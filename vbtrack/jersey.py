"""Jersey-number voting per player track.

A single read is unreliable (back turned, occlusion, blur), so we only count confident reads and assign a
number to a track once a few of them agree. The reader's model card: keep reads with confidence >= 0.85;
two or three agreeing reads identify the player reliably.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class JerseyVoter:
    def __init__(self, min_conf: float = 0.85, min_votes: int = 2):
        self.min_conf = min_conf
        self.min_votes = min_votes
        self.votes: dict[int, Counter] = defaultdict(Counter)

    def add(self, track_id: int, number: str | None, conf: float) -> None:
        if number is not None and conf >= self.min_conf:
            self.votes[track_id][number] += 1

    def number(self, track_id: int) -> str | None:
        """Leading number for the track, once it has `min_votes` reads and a clear lead over the runner-up."""
        c = self.votes.get(track_id)
        if not c:
            return None
        (top, n), *rest = c.most_common(2)
        if n < self.min_votes or (rest and rest[0][1] == n):
            return None
        return top

    def summary(self) -> dict[int, dict]:
        """{track_id: {"number", "votes": {number: count}}} for every track that got a confident read."""
        return {tid: {"number": self.number(tid), "votes": dict(c)} for tid, c in self.votes.items()}
