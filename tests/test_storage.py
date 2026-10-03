"""Storage tests. Uses a real MongoDB if MONGO_URI is set (needed for $merge), else mongomock."""
import os

import pytest

from vbtrack.storage import MongoStore


@pytest.fixture
def store():
    uri = os.getenv("MONGO_URI")
    if uri:
        s = MongoStore(uri, "vbtrack_test", batch_size=3)
        s.client.drop_database("vbtrack_test")
        s.ensure_indexes()
    else:
        mongomock = pytest.importorskip("mongomock")
        s = MongoStore(db="vbtrack_test", batch_size=3, client=mongomock.MongoClient())
    yield s
    if uri:
        s.client.drop_database("vbtrack_test")
    s.close()


def frame(i, players, ball=None):
    return {"frame": i, "t": i / 30, "players": players, "ball": ball}


def player(tid, x, y):
    return {"track_id": tid, "conf": 0.8, "bbox": [x - 20, y - 80, x + 20, y],
            "foot": {"x": x, "y": y}}


def test_batches_and_counts(store):
    sid = store.start_session("test.mp4", 30, 1920, 1080, live=False)
    for i in range(7):
        store.add_frame(frame(i, [player(1, 100, 500)]))
    assert store.db.frames.count_documents({"session_id": sid}) == 6   # two full batches
    store.flush()
    assert store.db.frames.count_documents({"session_id": sid}) == 7
    assert store.db.sessions.find_one({"_id": sid})["frame_count"] == 7


@pytest.mark.skipif(not os.getenv("MONGO_URI"), reason="$merge needs a real MongoDB")
def test_build_tracks(store):
    sid = store.start_session("test.mp4", 30, 1920, 1080, live=False)
    for i in range(10):
        ps = [player(1, 100 + i, 500)]
        if i >= 5:
            ps.append(player(2, 800, 400))
        store.add_frame(frame(i, ps, {"x": 1.0, "y": 2.0, "interpolated": False}))
    store.end_session()
    tracks = {t["track_id"]: t for t in store.db.tracks.find({"session_id": sid})}
    assert tracks[1]["frames_seen"] == 10 and tracks[1]["first_frame"] == 0
    assert tracks[2]["frames_seen"] == 5 and tracks[2]["first_frame"] == 5
    assert tracks[2]["avg_foot"] == {"x": 800, "y": 400}
    assert store.db.sessions.find_one({"_id": sid})["status"] == "done"
