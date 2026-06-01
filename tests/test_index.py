"""Tests for the SQLite ticket index: persistence, search ordering, rebuild."""

from __future__ import annotations

from operations_agent.embeddings import FakeEmbeddingClient
from operations_agent.index import IndexedTicket, TicketIndex


def test_search_ranks_more_similar_ticket_higher(tmp_path):
    idx = TicketIndex(tmp_path / "i.db", FakeEmbeddingClient())
    idx.add(IndexedTicket("A", "login crash on submit",
                          "app crashes when logging in"))
    idx.add(IndexedTicket("B", "export csv empty",
                          "csv export has no rows"))

    hits = idx.search("crash when I log in to the app", top_k=2)

    assert hits[0].key == "A"  # the login-crash ticket ranks first
    assert hits[0].score >= hits[1].score


def test_add_persists_and_upserts(tmp_path):
    path = tmp_path / "i.db"
    idx = TicketIndex(path, FakeEmbeddingClient())
    idx.add(IndexedTicket("A", "first", "first body"))
    idx.add(IndexedTicket("A", "updated", "updated body"))  # same key -> upsert
    assert idx.count() == 1
    idx.close()

    # Reopen: data persisted across connections.
    idx2 = TicketIndex(path, FakeEmbeddingClient())
    assert idx2.count() == 1
    assert idx2.search("updated", top_k=1)[0].summary == "updated"


def test_rebuild_replaces_contents(tmp_path):
    idx = TicketIndex(tmp_path / "i.db", FakeEmbeddingClient())
    idx.add(IndexedTicket("OLD", "old", "old"))
    n = idx.rebuild([
        IndexedTicket("X", "x", "x body"),
        IndexedTicket("Y", "y", "y body"),
    ])
    assert n == 2
    assert idx.count() == 2
    assert {h.key for h in idx.search("x body", top_k=5)} == {"X", "Y"}
