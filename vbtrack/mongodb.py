"""Simple PyMongo access for event documents."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.results import InsertOneResult


class MongoDB:
    def __init__(self, uri: str | None = None,
                 db: str | None = None, collection: str | None = None,
                 client: MongoClient | None = None):
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        uri = uri if uri is not None else os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db = db if db is not None else os.getenv("MONGO_DB", "vball")
        collection = collection if collection is not None else os.getenv("MONGO_COLLECTION", "events")
        self.client = client if client is not None else MongoClient(
            uri, serverSelectionTimeoutMS=5000
        )
        self.collection = self.client[db][collection]

    def write_data(self, document: dict[str, Any]) -> InsertOneResult:
        """Insert the supplied document and return the insertion result."""
        return self.collection.insert_one(document)

    def get_player_events(self, player_id: str) -> list[dict[str, Any]]:
        """Retrieve events for a player (not yet implemented)."""
        raise NotImplementedError()
