"""Chat history: local SQLite, no services.

Single-user local app, so each operation opens its own connection — no pool,
no locking games. The DB file lives at the project root (gitignored).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import PROJECT_ROOT

TITLE_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);
"""


class HistoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or PROJECT_ROOT / "history.db"
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_chat(self, title: str = "New chat") -> dict:
        now = self._now()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return {"id": cursor.lastrowid, "title": title, "created_at": now,
                    "updated_at": now, "turns": 0}

    def list_chats(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT c.*, COUNT(t.id) AS turns FROM chats c
                   LEFT JOIN turns t ON t.chat_id = c.id
                   GROUP BY c.id ORDER BY c.updated_at DESC"""
            ).fetchall()
            return [dict(row) for row in rows]

    def chat_exists(self, chat_id: int) -> bool:
        with self._connect() as db:
            return db.execute(
                "SELECT 1 FROM chats WHERE id = ?", (chat_id,)
            ).fetchone() is not None

    def get_turns(self, chat_id: int) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM turns WHERE chat_id = ? ORDER BY id", (chat_id,)
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": json.loads(row["answer_json"]),
                    "evidence": json.loads(row["evidence_json"])
                    if row["evidence_json"] else None,
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def add_turn(
        self,
        chat_id: int,
        question: str,
        answer_json: str,
        evidence_json: str | None = None,
    ) -> None:
        now = self._now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO turns (chat_id, question, answer_json, evidence_json, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (chat_id, question, answer_json, evidence_json, now),
            )
            # first turn titles the chat after the question
            (count,) = db.execute(
                "SELECT COUNT(*) FROM turns WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if count == 1:
                title = question[:TITLE_CHARS].strip() or "New chat"
                db.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
            db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    def delete_chat(self, chat_id: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def recent_history(self, chat_id: int, n: int = 4) -> list[tuple[str, str]]:
        """Last n (question, answer_text) pairs, oldest first — prompt-ready."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT question, answer_json FROM turns WHERE chat_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (chat_id, n),
            ).fetchall()
        return [
            (row["question"], json.loads(row["answer_json"])["text"])
            for row in reversed(rows)
        ]
