import torch

from vbtrack.jersey import JerseyVoter
from vbtrack.models import decode_numbers, merge_tiles, tiles_for


def test_voter_needs_agreeing_confident_reads():
    v = JerseyVoter(min_conf=0.85, min_votes=2)
    v.add(1, "12", 0.95)
    assert v.number(1) is None          # one read is not enough
    v.add(1, "17", 0.60)                # low confidence: ignored
    v.add(1, None, 0.99)                # unreadable: ignored
    v.add(1, "12", 0.90)
    assert v.number(1) == "12"
    assert v.summary()[1] == {"number": "12", "votes": {"12": 2}}


def test_voter_tie_is_unresolved():
    v = JerseyVoter(min_votes=2)
    for n in ("4", "4", "14", "14"):
        v.add(7, n, 0.9)
    assert v.number(7) is None
    assert v.number(99) is None


def test_decode_numbers():
    logits = torch.full((3, 23), -10.0)
    logits[0, 1], logits[0, 11 + 2], logits[0, 22] = 10, 10, 10    # tens=1, ones=2, readable -> "12"
    logits[1, 10], logits[1, 11 + 7], logits[1, 22] = 10, 10, 10   # blank tens, ones=7 -> "7"
    logits[2, 21] = 10                                              # unreadable
    (n0, c0), (n1, _), (n2, _) = decode_numbers(logits)
    assert (n0, n1, n2) == ("12", "7", None)
    assert c0 > 0.99


def test_tiles_cover_frame_and_include_full_pass():
    tiles = tiles_for(1920, 1080, 864)
    assert tiles[-1] == (0, 0, 1920, 1080)
    assert max(t[2] for t in tiles[:-1]) == 1920 and max(t[3] for t in tiles[:-1]) == 1080
    assert tiles_for(800, 600, 864) == [(0, 0, 800, 600)]


def test_merge_drops_duplicates_and_covered_cut_boxes():
    dets = [
        (100, 100, 120, 120, 0.9, 1, False),
        (101, 101, 121, 121, 0.7, 1, False),   # duplicate from an overlapping tile
        (100, 100, 115, 120, 0.8, 1, True),    # cut copy of the same ball
        (500, 500, 520, 520, 0.6, 1, True),    # cut, but nothing else covers it -> kept
    ]
    kept = merge_tiles(dets)
    assert sorted(round(d[4], 1) for d in kept) == [0.6, 0.9]
