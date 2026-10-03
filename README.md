# vbtrack: volleyball player + ball tracking into MongoDB

Tracks players (with persistent IDs) and the ball in recorded video or live streams, then writes every frame to MongoDB.

## How it works

```
video file / webcam / RTSP
        │  OpenCV capture (auto-reconnect for live streams)
        ▼
YOLO detector (one pass: person + ball)          optional: custom volleyball model
        │                         │                       │
        ▼                         ▼                       ▼
ByteTrack / BoT-SORT        ball candidates  ◄────────────┘
(persistent player IDs)           │
        │                   BallTracker: Kalman filter + gating
        │                   (one ball, rejects false hits, fills short gaps)
        ▼                         ▼
   court ROI filter  ──►  frame document  ──►  MongoDB (batched insert_many)
                                                   └─ on finish: $merge → tracks
```

**Players.** YOLO's COCO `person` class does well here. The tracker runs on person boxes only and gives each player a `track_id`. Pass a court polygon (`--roi`) to drop spectators, referees on the bench and so on. A player counts as on court when their foot point (bottom-centre of the box) is inside the polygon.

**Ball.** This is the hard part. A volleyball is small and blurred, and it often sits against busy backgrounds.
- The stock COCO `sports ball` class works, but detections often come with low confidence. That's why the defaults are `--ball-conf 0.15` and `--imgsz 1280`.
- The ball is *not* sent through ByteTrack, because that would silently drop low-confidence ball detections. A dedicated `BallTracker` handles it instead. It gates candidates by distance to the Kalman-predicted position, coasts through up to 10 missed frames (stored with `interpolated: true`), and re-acquires the ball after losing it.
- For serious accuracy, fine-tune a ball model on a volleyball dataset (Roboflow Universe has several) and pass it with `--ball-weights best.pt --ball-class 0`.

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

# custom ball model, faster on CPU by processing every 2nd frame
python -m vbtrack run --source match.mp4 --ball-weights volleyball.pt --stride 2 --device cpu
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
  frame_count, player_weights, ball_weights, tracker, stride, roi }
```

**frames**: one per processed frame; unique index on `(session_id, frame)`
```js
{ session_id, frame: 1234, t: 41.133,           // seconds from start
  ts: ISODate(...),                             // wall clock (exact for live streams)
  players: [ { track_id: 3, conf: 0.87, bbox: [x1,y1,x2,y2], foot: {x, y} } ],
  ball: { x, y, vx, vy, conf, bbox, interpolated } | null }
```

**tracks**: one per player track; rebuilt with `$merge` when a session ends
```js
{ session_id, track_id, first_frame, last_frame, first_t, last_t,
  frames_seen, avg_conf, avg_foot: {x, y} }
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
| Ball missed a lot | custom `--ball-weights`; `--imgsz 1280` or more; lower `--ball-conf` |
| Ball jumps onto heads or lights | raise `--ball-conf`; lower `gate_px` in `BallTracker` |
| Player IDs swap after occlusion | `--tracker botsort.yaml` |
| Too slow | `yolo11n`/`yolo26n`, `--imgsz 960`, `--stride 2`, GPU (`--device 0`) or Apple `mps` |
| Spectators tracked | `--roi` court polygon |

Pixel coordinates depend on the camera. To get court coordinates in metres, compute a homography from the 4 court corners (`cv2.findHomography`) and transform `foot` and the ball positions.

## Tests

```bash
pytest                                       # ball tracker + storage (mongomock)
MONGO_URI=mongodb://localhost:27017 pytest   # also runs the $merge track summary against real Mongo
```
