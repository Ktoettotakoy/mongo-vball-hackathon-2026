import mongomock
import numpy as np
import pytest

from vbtrack.label import contact_sheets, extract_json, normalise_events
from vbtrack.stats import player_stats


def test_extract_json_fenced_and_embedded():
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('Here you go: [{"a": 1}] done') == [{"a": 1}]
    assert extract_json('{"events": []}') == {"events": []}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_normalise_events_cleans_and_drops():
    raw = [
        {"frame": "42", "track_id": "19", "player_no": "7", "action": "Attack", "successful": "true"},
        {"track_id": "#3", "action": "block", "successful": False, "block_type": "full"},
        {"track_id": "#3", "action": "serve", "successful": True, "block_type": "full"},
        {"track_id": "", "action": "serve"},            # no id -> dropped
        {"track_id": "#5", "action": "dig"},            # unknown action -> dropped
        "junk",
    ]
    events, dropped = normalise_events({"events": raw})
    assert dropped == 3
    assert events[0] == {"frame": 42, "track_id": "#19", "player_no": 7, "action": "spike", "successful": True}
    assert events[1]["block_type"] == "full"
    assert "block_type" not in events[2]


def test_contact_sheets_respects_attachment_limit():
    frames = [(i, np.zeros((360, 640, 3), dtype=np.uint8)) for i in range(60)]
    sheets = contact_sheets(frames, sheets=10)
    assert len(sheets) == 10
    assert all(s[:2] == b"\xff\xd8" for s in sheets)   # JPEG magic


def test_player_stats_groups_by_track_id():
    coll = mongomock.MongoClient().db.events
    coll.insert_many([
        {"video": "a.mp4", "track_id": "#1", "player_no": None, "action": "spike", "successful": True},
        {"video": "a.mp4", "track_id": "#1", "player_no": 9, "action": "spike", "successful": False},
        {"video": "b.mp4", "track_id": "#1", "player_no": None, "action": "serve", "successful": True},
        {"video": "a.mp4", "track_id": "#2", "player_no": None, "action": "block", "successful": True,
         "block_type": "solo"},
    ])
    rows = {r["track_id"]: r for r in player_stats(coll)}
    p1 = rows["#1"]
    assert (p1["total"], p1["spikes"], p1["kills"], p1["serves"], p1["serves_won"]) == (3, 2, 1, 1, 1)
    assert p1["player_no"] == 9
    assert rows["#2"]["blocks_won"] == 1 and rows["#2"]["solo_blocks"] == 1
    assert {r["track_id"] for r in player_stats(coll, video="b.mp4")} == {"#1"}
