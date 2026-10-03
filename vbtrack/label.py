"""Label volleyball actions with an LLM (via Hoplite) and store them in MongoDB.

For each video: sample N frames (default 60), tile them into contact sheets, send them
with a prompt to a Hoplite agent thread, wait for the JSON reply, validate it, and write
one document per action to the `events` collection. When every video is done, the
per-player aggregate (vbtrack/stats.py) is rebuilt and printed.

  python -m vbtrack.label annotated_7.mp4 annotated_8.mp4
  python -m vbtrack.label game1.mp4 --session <session_id>   # draw #track_id boxes from Mongo `frames`
  python -m vbtrack.label annotated_*.mp4 --force            # re-label videos already done

Environment (.env in the repo root):
  Token_for_LLM       Hoplite API key (HOPLITE_API_KEY also works)
  HOPLITE_PROJECT_ID  optional; a project named "vbtrack-labelling" is created when absent
  HOPLITE_MODEL       optional; default is the workspace default model
  MONGO_URI, MONGO_DB, MONGO_COLLECTION
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from bson import ObjectId
from dotenv import load_dotenv

from .mongodb import MongoDB
from .stats import player_stats, print_table, refresh_player_stats

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
log = logging.getLogger("vbtrack.label")

API_URL = os.getenv("HOPLITE_API_URL", "https://api.hoplite.sh")
MAX_ATTACHMENTS = 10          # Hoplite limit per thread creation
ACTIONS = {"serve", "hit", "spike", "block"}
BLOCK_TYPES = {"partial", "full"}
TERMINAL = {"completed", "failed", "cancelled", "canceled", "error", "stopped"}
BLOCKED = {"waiting_for_approval", "awaiting_approval", "paused", "blocked"}

# Team prompt. Only the first paragraph (frames come as contact sheets, not a video file)
# and the "do not run tools" line are added. Braces are doubled for str.format.
PROMPT = """The attached images are contact sheets of {n_frames} frames from one annotated volleyball clip
("{video}"), in time order. Read the tiles left to right, top to bottom, sheet by sheet. Each tile
shows its frame number in the top-left corner. Do not edit files or run commands. Only look at the images.

Watch this annotated volleyball clip. Boxes and red labels are on players. A label is a jersey number, or a track id like #4 if the number was not read. The ball may have an orange circle.

Return only a JSON array, one object per ball contact you actually see. Do not invent contacts that happen off-screen or after the clip ends.

Each object:
{{
  "player_no": "<jersey number, or null if only a track id or no label>",
  "track_id": "<the # label if there is no jersey number, else null>",
  "action": "serve" | "hit" | "spike" | "block",
  "successful": true | false
}}

If action is "block", also include "block_type": "partial" | "full".
Partial means the ball still crosses or goes past the block. Full means the ball stays on the attacker's side.

spike is an attack jump-hit. hit is any other attack contact that is not a spike. If you cannot tell spike from hit, use "hit".

successful is true only when that contact does its job: a serve or attack that stays in play for the other side, or a block that touches the ball. If the ball is out, in the net, or the contact misses, successful is false.

Reply with the JSON array only. No prose, no markdown fences.
"""


# ---------------------------------------------------------------- frames
def sample_frames(path: str, n: int) -> tuple[list[tuple[int, np.ndarray]], float]:
    """Return n evenly spaced (frame_index, image) pairs and the video fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        raise RuntimeError(f"video has no frames: {path}")
    wanted = sorted(set(np.linspace(0, total - 1, num=min(n, total), dtype=int).tolist()))
    out = []
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if ok:
            out.append((idx, img))
    cap.release()
    return out, fps


def draw_tracks(frames: list[tuple[int, np.ndarray]], frame_docs: dict[int, dict]) -> None:
    """Draw #track_id boxes from Mongo `frames` docs onto raw frames (in place)."""
    for idx, img in frames:
        doc = frame_docs.get(idx)
        if not doc:
            continue
        for p in doc.get("players", []):
            x1, y1, x2, y2 = map(int, p["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, f"#{p['track_id']}", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def contact_sheets(frames: list[tuple[int, np.ndarray]], sheets: int = MAX_ATTACHMENTS,
                   tile_w: int = 640, cols: int = 2) -> list[bytes]:
    """Tile frames into at most `sheets` JPEG images, each tile labelled with its frame number."""
    per_sheet = max(1, -(-len(frames) // sheets))   # ceil
    rows = -(-per_sheet // cols)
    out = []
    for s in range(0, len(frames), per_sheet):
        chunk = frames[s:s + per_sheet]
        tiles = []
        for idx, img in chunk:
            h, w = img.shape[:2]
            t = cv2.resize(img, (tile_w, int(h * tile_w / w)), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (110, 34), (0, 0, 0), -1)
            cv2.putText(t, f"f{idx}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(t)
        blank = np.zeros_like(tiles[0])
        tiles += [blank] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out.append(buf.tobytes())
    return out


# ---------------------------------------------------------------- hoplite
class Hoplite:
    def __init__(self, api_key: str, base_url: str = API_URL, timeout: float = 60):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"X-Api-Key": api_key, "Accept": "application/json"})

    def _req(self, method: str, path: str, idem: str | None = None, **kw) -> dict:
        headers = {"Idempotency-Key": idem} if idem else {}
        r = self.s.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Hoplite {method} {path}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}

    def default_model(self) -> str | None:
        data = self._req("GET", "/api/model-providers")
        return (data.get("runConfig") or {}).get("defaultModelId")

    def ensure_project(self, project_id: str | None, name: str = "vbtrack-labelling") -> str:
        if project_id:
            return project_id
        data = self._req("GET", "/api/projects")
        items = data.get("projects") or data.get("items") or data.get("data") or []
        for p in items:
            if p.get("name") == name:
                return p["id"]
        data = self._req("POST", "/api/projects", idem=f"vbtrack-project-{name}", json={"name": name})
        pid = (data.get("project") or data).get("id")
        if not pid:
            raise RuntimeError(f"project creation returned no id: {data}")
        log.info("created Hoplite project %s (set HOPLITE_PROJECT_ID=%s to reuse it)", name, pid)
        return pid

    def upload(self, project_id: str, op_id: str, filename: str, data: bytes, ctype: str) -> str:
        meta = self._req("POST", "/api/threads/initial/attachments/presign",
                         idem=f"{op_id}-{filename}", params={"projectId": project_id},
                         json={"initialClientOperationId": op_id, "filename": filename,
                               "contentType": ctype, "byteSize": len(data)})
        up = meta["upload"]
        # Presigned URL: never forward the Hoplite credential.
        r = requests.put(up["uploadUrl"], data=data, timeout=self.timeout,
                         headers={"Content-Type": up.get("contentType", ctype)})
        r.raise_for_status()
        return up["ticket"]

    def create_thread(self, project_id: str, op_id: str, prompt: str, tickets: list[str],
                      model: str | None, spend_limit_micros: int) -> str:
        body: dict[str, Any] = {"projectId": project_id, "prompt": prompt, "clientOperationId": op_id,
                                "initialAttachmentTickets": tickets, "spendLimitMicros": spend_limit_micros}
        if model:
            body["model"] = model
        data = self._req("POST", "/api/threads", idem=op_id, json=body)
        return data["thread"]["id"]

    def wait_reply(self, thread_id: str, timeout_s: float, poll_s: float = 5) -> str:
        deadline = time.monotonic() + timeout_s
        status = None
        while time.monotonic() < deadline:
            run = self._req("GET", f"/api/threads/{thread_id}/run-state").get("run") or {}
            status = str(run.get("status", "")).lower()
            if status in TERMINAL:
                if status != "completed":
                    raise RuntimeError(f"thread {thread_id} ended with status {status}")
                msg = (run.get("assistantMessage") or {}).get("content")
                return msg if msg else self.last_assistant_message(thread_id)
            if status in BLOCKED:
                raise RuntimeError(f"thread {thread_id} waits for a tool approval; resolve it in Hoplite")
            time.sleep(poll_s)
        raise TimeoutError(f"thread {thread_id} still '{status}' after {timeout_s:.0f}s")

    def last_assistant_message(self, thread_id: str) -> str:
        data = self._req("GET", f"/api/threads/{thread_id}/messages")
        items = data.get("messages") or data.get("items") or data.get("data") or []
        replies = [m for m in items if m.get("role") == "assistant" and m.get("kind", "chat") == "chat"]
        if not replies:
            raise RuntimeError(f"thread {thread_id} has no assistant reply")
        content = replies[-1].get("content")
        return content if isinstance(content, str) else json.dumps(content)


# ---------------------------------------------------------------- parsing
def extract_json(text: str) -> Any:
    """Parse the reply as JSON. Accept fenced blocks or JSON embedded in prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"reply is not JSON: {text[:300]!r}")


def _to_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "success", "successful"}
    return bool(v)


def normalise_events(data: Any) -> tuple[list[dict], int]:
    """Validate LLM items. Return (clean events, number of dropped items)."""
    if isinstance(data, dict):
        data = data.get("events") or data.get("actions") or []
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of events")
    clean, dropped = [], 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        action = str(item.get("action", "")).strip().lower()
        action = {"attack": "spike", "kill": "spike"}.get(action, action)
        tid = str(item.get("track_id") or item.get("trackId") or "").strip()
        digits = re.sub(r"\D", "", tid)
        if action not in ACTIONS or not digits:
            dropped += 1
            continue
        pno = item.get("player_no", item.get("playerNo"))
        try:
            pno = int(pno) if pno not in (None, "") else None
        except (TypeError, ValueError):
            pno = None
        frame = item.get("frame")
        try:
            frame = int(frame) if frame is not None else None
        except (TypeError, ValueError):
            frame = None
        block_type = item.get("block_type", item.get("blockType"))
        ev = {"frame": frame, "track_id": f"#{int(digits)}", "player_no": pno,
              "action": action, "successful": _to_bool(item.get("successful"))}
        if action == "block" and block_type in ("full", "solo"):
            ev["block_type"] = block_type
        clean.append(ev)
    return clean, dropped


# ---------------------------------------------------------------- mongo
def load_frame_docs(db, session_id: str, indices: list[int]) -> dict[int, dict]:
    cur = db["frames"].find({"session_id": ObjectId(session_id), "frame": {"$in": indices}},
                            {"_id": 0, "frame": 1, "players": 1})
    return {d["frame"]: d for d in cur}


def store_events(mongo: MongoDB, video: str, events: list[dict], run_meta: dict) -> None:
    now = datetime.now(timezone.utc)
    coll = mongo.collection
    coll.create_index([("video", 1), ("frame", 1)])
    coll.create_index("track_id")
    coll.delete_many({"video": video})        # a re-label replaces the old events
    if events:
        coll.insert_many([{**e, "video": video, "labelled_at": now,
                           "thread_id": run_meta.get("thread_id"), "source": "llm"} for e in events])
    coll.database["label_runs"].update_one(
        {"video": video},
        {"$set": {**run_meta, "video": video, "n_events": len(events), "status": "done", "labelled_at": now}},
        upsert=True,
    )


def mark_failed(mongo: MongoDB, video: str, err: str, run_meta: dict) -> None:
    mongo.collection.database["label_runs"].update_one(
        {"video": video},
        {"$set": {**run_meta, "video": video, "status": "failed", "error": err,
                  "labelled_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


# ---------------------------------------------------------------- main
def label_video(path: str, args, hop: Hoplite, mongo: MongoDB, project_id: str, model: str | None) -> int:
    video = Path(path).name
    frames, fps = sample_frames(path, args.frames)
    if args.session:
        draw_tracks(frames, load_frame_docs(mongo.collection.database, args.session, [i for i, _ in frames]))
    sheets = contact_sheets(frames, sheets=min(args.sheets, MAX_ATTACHMENTS))
    if args.dump_dir:
        out = Path(args.dump_dir) / Path(video).stem
        out.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(sheets):
            (out / f"sheet_{i:02d}.jpg").write_bytes(b)

    op_id = f"vbtrack-{Path(video).stem}-{uuid.uuid4().hex[:8]}"
    meta = {"model": model, "frames_sent": len(frames), "sheets": len(sheets), "session_id": args.session}
    log.info("%s: %d frames -> %d sheets, uploading", video, len(frames), len(sheets))
    try:
        tickets = [hop.upload(project_id, op_id, f"sheet_{i:02d}.jpg", b, "image/jpeg")
                   for i, b in enumerate(sheets)]
        prompt = PROMPT.format(n_frames=len(frames), video=video, fps=fps)
        thread_id = hop.create_thread(project_id, op_id, prompt, tickets, model, args.spend_limit_micros)
        meta["thread_id"] = thread_id
        log.info("%s: thread %s started, waiting for reply", video, thread_id)
        reply = hop.wait_reply(thread_id, args.timeout)
        meta["raw_response"] = reply
        events, dropped = normalise_events(extract_json(reply))
    except Exception as e:  # keep going with the other videos
        log.error("%s: %s", video, e)
        mark_failed(mongo, video, str(e), meta)
        return -1
    store_events(mongo, video, events, {**meta, "dropped": dropped})
    log.info("%s: stored %d events (%d invalid items dropped)", video, len(events), dropped)
    return len(events)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vbtrack.label", description="LLM action labelling via Hoplite -> MongoDB")
    ap.add_argument("videos", nargs="+", help="video files (annotated videos show #track_id labels)")
    ap.add_argument("--frames", type=int, default=60, help="frames sampled per video")
    ap.add_argument("--sheets", type=int, default=MAX_ATTACHMENTS, help="contact sheets per video (max 10)")
    ap.add_argument("--session", default=None, help="Mongo session id: draw #track_id boxes from `frames`")
    ap.add_argument("--model", default=os.getenv("HOPLITE_MODEL"))
    ap.add_argument("--project", default=os.getenv("HOPLITE_PROJECT_ID"))
    ap.add_argument("--spend-limit-micros", type=int, default=int(os.getenv("HOPLITE_SPEND_LIMIT_MICROS", "1000000")))
    ap.add_argument("--timeout", type=float, default=900, help="seconds to wait for each reply")
    ap.add_argument("--force", action="store_true", help="re-label videos that already have a done run")
    ap.add_argument("--dump-dir", default=None, help="also save the contact sheets here (debug)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mongo = MongoDB()   # also loads .env
    api_key = os.getenv("Token_for_LLM") or os.getenv("HOPLITE_API_KEY")
    if not api_key:
        sys.exit("Set Token_for_LLM (or HOPLITE_API_KEY) in .env")
    hop = Hoplite(api_key)
    try:
        project_id = hop.ensure_project(args.project)
        model = args.model or hop.default_model()
        runs = mongo.collection.database["label_runs"]
        done = 0
        for path in args.videos:
            video = Path(path).name
            if not args.force and runs.find_one({"video": video, "status": "done"}):
                log.info("%s: already labelled, skipping (use --force)", video)
                continue
            if label_video(path, args, hop, mongo, project_id, model) >= 0:
                done += 1

        log.info("labelling finished: %d/%d videos labelled now", done, len(args.videos))
        n = refresh_player_stats(mongo.collection)
        print()
        print_table(player_stats(mongo.collection))
        print(f"\nplayer_stats: {n} players")
    finally:
        mongo.client.close()


if __name__ == "__main__":
    main()
