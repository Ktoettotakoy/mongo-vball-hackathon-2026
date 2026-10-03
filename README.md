# vbtrack: volleyball player + ball tracking into MongoDB

Tracks players (with persistent IDs) and the ball in recorded video or live streams, then writes every frame to MongoDB.

## Models

All three come from the Hugging Face Hub. They download on first run and are cached in `~/.cache/huggingface`.

| job | model | notes |
|-----|-------|-------|
| people + ball | [grdesignbuild/volleyball-person-ball-detector](https://huggingface.co/grdesignbuild/volleyball-person-ball-detector) | D-FINE-S fine-tuned on volleyball (`person`, `ball`). The default detector. |
| jersey numbers | [grdesignbuild/volleyball-jersey-number-reader](https://huggingface.co/grdesignbuild/volleyball-jersey-number-reader) | ResNet-18 that reads the number on a torso crop, or says it's unreadable |
| 2nd ball model + court | [Davidsv/volley-ref-ai](https://huggingface.co/Davidsv/volley-ref-ai) | YOLO11s ball detector (extra candidates for the ball tracker) and YOLO11n-pose court keypoints (automatic court ROI) |

## How it works

```
video file / webcam / RTSP
        │  OpenCV capture (auto-reconnect for live streams)
        ▼
D-FINE volleyball detector ── person boxes ──► ByteTrack / BoT-SORT (persistent player IDs)
        │                                              │
        └─ ball boxes ─┐                               ├─► jersey reader on torso crops ─► per-track number vote
volley-ref YOLO ball ──┴► BallTracker: Kalman + gating │
                          (one ball, fills short gaps) │
volley-ref court keypoints ─► court ROI ───────────────┘ (drops spectators / bench)
                                   ▼
                  frame document ──► MongoDB (batched insert_many)
                                      └─ on finish: $merge → tracks (+ jersey number)
```

**Players.** The fine-tuned D-FINE model finds people. The tracker gives each one a `track_id`. A player counts as on court when their foot point (bottom-centre of the box) is inside the court polygon.

**Court.** Without `--roi`, the volley-ref court keypoint model is sampled every 15 processed frames until it has 5 good views. The convex hull of the median keypoints, grown 15%, becomes the ROI and is saved on the session (`roi`, `court.keypoints`). A view counts only when its polygon contains at least 4 players' feet. On views the court model wasn't trained for (wide arena shots, broadcast close-ups), it returns confident but wrong keypoints, and this check keeps them out. When no view passes, no ROI is applied. The camera is assumed to be static.

**Jersey numbers.** Every 3rd processed frame (`--jersey-every`), the reader looks at the chest of every player at least 70 px tall. Readable numbers are stored on the player as `jersey_read`. Reads with confidence ≥ 0.85 count as votes for that track. Once two votes agree, the track gets a `number`, which is written on every later frame and on the `tracks` document. Numbers are often unreadable (back turned, far away), so expect most frames to have none.

**Ball.** A volleyball is small and blurred, and it often sits against busy backgrounds.
- D-FINE's `ball` class is the main source (`--ball-conf 0.4`). The volley-ref YOLO ball model adds candidates above `--ref-ball-conf 0.7`. Below that it fires on signs and lights. Turn it off with `--ball-weights none`.
- The ball is *not* sent through ByteTrack. A dedicated `BallTracker` gates candidates by distance to the Kalman-predicted position, coasts through up to 10 missed frames (stored with `interpolated: true`), and re-acquires the ball after losing it.
- By default, frames up to 2000 px wide go through D-FINE in one pass (about 4 fps at 1080p on Apple MPS). 4K frames are cut into 864 px tiles, as the model card recommends. `--tile 864` also tiles 1080p, which catches smaller far-court balls but is about 5× slower.

`--detector yolo` switches back to the stock COCO YOLO person / sports-ball detector (`--player-weights`).

## Quick start

```bash
docker compose up -d                 # MongoDB on :27017, mongo-express UI on :8081
pip install -r requirements.txt

# recorded match
python -m vbtrack run --source match.mp4 --save-video annotated.mp4

# live: webcam or IP camera
python -m vbtrack run --source 0 --show
python -m vbtrack run --source rtsp://user:pass@192.168.1.20/stream1 --show

# keep only players on court (pixel coords of the court corners)
python -m vbtrack run --source match.mp4 --roi "120,300;1800,300;1900,1050;20,1050"

# faster: process every 2nd frame, skip the second ball model
python -m vbtrack run --source match.mp4 --stride 2 --ball-weights none

# your own ball model (local .pt or hf://owner/repo/file.pt)
python -m vbtrack run --source match.mp4 --ball-weights volleyball.pt --ref-ball-conf 0.5
```

Connection settings: `--mongo` / `--db`, or the `MONGO_URI` / `MONGO_DB` environment variables.

## Querying

```bash
python -m vbtrack sessions
python -m vbtrack players --session <id>
python -m vbtrack ball --session <id> --csv ball.csv --detected-only
```

## MongoDB schema

**sessions**: one per run
```js
{ _id, source: "match.mp4", live: false, fps: 30, width: 1920, height: 1080,
  started_at, ended_at, status: "running" | "done" | "stopped" | "stream_lost",
  frame_count, detector, player_weights, ball_weights, jersey_weights, tracker, stride,
  roi,                                          // given with --roi, or found by the court model
  court: { source, keypoints: [ {x, y, seen} | null ] } }   // 14 court keypoints, when auto-detected
```

**frames**: one per processed frame; unique index on `(session_id, frame)`
```js
{ session_id, frame: 1234, t: 41.133,           // seconds from start
  ts: ISODate(...),                             // wall clock (exact for live streams)
  players: [ { track_id: 3, conf: 0.87, bbox: [x1,y1,x2,y2], foot: {x, y},
               number: "12" | null,                       // voted jersey number of this track so far
               jersey_read: { number: "12", conf: 0.91 } } ],  // this frame's raw read, when readable
  ball: { x, y, vx, vy, conf, bbox, interpolated } | null }
```

**tracks**: one per player track; rebuilt with `$merge` when a session ends
```js
{ session_id, track_id, first_frame, last_frame, first_t, last_t,
  frames_seen, avg_conf, avg_foot: {x, y},
  number: "12" | null, jersey_votes: { "12": 5, "17": 1 } }
```

Example queries in mongosh:
```js
// ball path for one rally window
db.frames.find({session_id: ObjectId("..."), t: {$gte: 60, $lte: 75}, ball: {$ne: null}},
               {frame: 1, t: 1, "ball.x": 1, "ball.y": 1})

// how often the ball was actually detected vs predicted
db.frames.aggregate([{$match: {session_id: ObjectId("...")}},
  {$group: {_id: "$ball.interpolated", n: {$sum: 1}}}])

// one player's movement (heatmap input)
db.frames.aggregate([{$match: {session_id: ObjectId("..."), "players.track_id": 3}},
  {$unwind: "$players"}, {$match: {"players.track_id": 3}},
  {$project: {t: 1, x: "$players.foot.x", y: "$players.foot.y"}}])
```

## Tuning tips

| Problem | Try |
|---|---|
| Ball missed a lot | `--tile 864` (tiles 1080p frames); lower `--ball-conf` / `--ref-ball-conf` |
| Ball jumps onto heads, signs or lights | raise `--ball-conf`; `--ball-weights none`; lower `gate_px` in `BallTracker` |
| Player IDs swap after occlusion | `--tracker botsort.yaml` |
| Too slow | `--stride 2`, `--ball-weights none`, `--jersey-every 10`, GPU (`--device 0`) |
| Spectators tracked | court not found automatically: pass `--roi` |
| No jersey numbers | the camera is too far or the view isn't like the training footage; numbers need ~70 px tall players and a visible chest |

Pixel coordinates depend on the camera. To get court coordinates in metres, compute a homography from the 4 court corners (`cv2.findHomography`) and transform `foot` and the ball positions.

## Tests

```bash
pytest                                       # ball tracker, jersey voting, tiling + storage (mongomock)
MONGO_URI=mongodb://localhost:27017 pytest   # also runs the $merge track summary against real Mongo
```
