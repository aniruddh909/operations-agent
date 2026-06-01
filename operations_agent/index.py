"""A persisted ticket index for semantic duplicate search.

Deliberately humble infrastructure: a SQLite table of (key, summary, text,
vector) plus an in-Python cosine scan. A managed vector DB is out of scope for
this project (see PRD); at a few dozen-to-hundred tickets a linear scan is
instant, and the simplicity keeps the interesting part — retrieve-then-adjudicate
— front and centre.

Vectors are produced on ingest (``add``) by an injected ``EmbeddingClient``, so
filing a ticket also indexes it. ``rebuild`` re-embeds a full set of tickets and
is what the ``reindex`` CLI command calls to recover from drift.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingClient, Vector, cosine


@dataclass
class IndexedTicket:
    key: str
    summary: str
    text: str


@dataclass
class SearchHit:
    key: str
    summary: str
    text: str
    score: float


class TicketIndex:
    """SQLite-backed store of ticket vectors with cosine top-K search."""

    def __init__(self, db_path: str | Path, embedder: EmbeddingClient) -> None:
        self._path = str(db_path)
        self._embedder = embedder
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                key     TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                text    TEXT NOT NULL,
                vector  TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- ingest -- #

    def add(self, ticket: IndexedTicket) -> None:
        """Embed and upsert one ticket (embed-on-ingest)."""
        [vector] = self._embedder.embed([_ticket_text(ticket)])
        self._upsert(ticket, vector)

    def rebuild(self, tickets: list[IndexedTicket]) -> int:
        """Drop all vectors and re-embed the given tickets. Returns the count."""
        self._conn.execute("DELETE FROM tickets")
        if tickets:
            vectors = self._embedder.embed([_ticket_text(t) for t in tickets])
            for ticket, vector in zip(tickets, vectors):
                self._upsert(ticket, vector, commit=False)
        self._conn.commit()
        return len(tickets)

    def _upsert(
        self, ticket: IndexedTicket, vector: Vector, commit: bool = True
    ) -> None:
        self._conn.execute(
            "INSERT INTO tickets (key, summary, text, vector) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET summary=excluded.summary, "
            "text=excluded.text, vector=excluded.vector",
            (ticket.key, ticket.summary, ticket.text, json.dumps(vector)),
        )
        if commit:
            self._conn.commit()

    # -- query -- #

    def search(self, query_text: str, *, top_k: int = 5) -> list[SearchHit]:
        """Return the top-K stored tickets by cosine similarity to the query."""
        [query_vec] = self._embedder.embed([query_text])
        hits: list[SearchHit] = []
        for key, summary, text, vector_json in self._conn.execute(
            "SELECT key, summary, text, vector FROM tickets"
        ):
            score = cosine(query_vec, json.loads(vector_json))
            hits.append(SearchHit(key, summary, text, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _ticket_text(ticket: IndexedTicket) -> str:
    return f"{ticket.summary}\n\n{ticket.text}".strip()
