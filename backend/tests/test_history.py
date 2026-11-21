"""HistoryStore: chats, turns, auto-titling, prompt history."""

import json

import pytest

from app.history import HistoryStore


@pytest.fixture
def store(tmp_path) -> HistoryStore:
    return HistoryStore(db_path=tmp_path / "history.db")


def answer_json(text: str) -> str:
    return json.dumps({"text": text, "citations": [], "used_chunk_ids": [],
                       "abstained": False, "kind": "grounded"})


class TestChats:
    def test_create_and_list(self, store):
        chat = store.create_chat()
        assert chat["title"] == "New chat"
        assert [c["id"] for c in store.list_chats()] == [chat["id"]]

    def test_first_turn_titles_chat(self, store):
        chat = store.create_chat()
        store.add_turn(chat["id"], "What torque does XK-42817 need?", answer_json("5.5 Nm"))
        assert store.list_chats()[0]["title"] == "What torque does XK-42817 need?"
        store.add_turn(chat["id"], "second question", answer_json("x"))
        assert store.list_chats()[0]["title"] == "What torque does XK-42817 need?"

    def test_most_recently_used_first(self, store):
        first = store.create_chat()
        second = store.create_chat()
        store.add_turn(first["id"], "q", answer_json("a"))
        assert [c["id"] for c in store.list_chats()] == [first["id"], second["id"]]

    def test_delete_cascades(self, store):
        chat = store.create_chat()
        store.add_turn(chat["id"], "q", answer_json("a"))
        store.delete_chat(chat["id"])
        assert store.list_chats() == []
        assert store.get_turns(chat["id"]) == []
        assert not store.chat_exists(chat["id"])


class TestTurns:
    def test_turns_round_trip(self, store):
        chat = store.create_chat()
        store.add_turn(chat["id"], "q1", answer_json("a1"),
                       evidence_json=json.dumps({"items": []}))
        turns = store.get_turns(chat["id"])
        assert turns[0]["question"] == "q1"
        assert turns[0]["answer"]["text"] == "a1"
        assert turns[0]["evidence"] == {"items": []}
        assert store.list_chats()[0]["turns"] == 1

    def test_recent_history_ordered_and_limited(self, store):
        chat = store.create_chat()
        for i in range(6):
            store.add_turn(chat["id"], f"q{i}", answer_json(f"a{i}"))
        history = store.recent_history(chat["id"], n=4)
        assert history == [(f"q{i}", f"a{i}") for i in range(2, 6)]  # oldest first
