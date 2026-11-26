from __future__ import annotations

import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, day TEXT, conv_key TEXT, requested_model TEXT,
  route TEXT, source TEXT, reason TEXT, backend TEXT, status INTEGER,
  latency_ms INTEGER, classifier_ms INTEGER,
  est_prompt_tokens INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
  cost_usd REAL
);
CREATE TABLE IF NOT EXISTS sticky (
  conv_key TEXT PRIMARY KEY, route TEXT, ts REAL
);
"""


def today() -> str:
    return time.strftime("%Y-%m-%d")


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def log_request(self, **f) -> None:
        f.setdefault("ts", time.time())
        f.setdefault("day", today())
        cols = ",".join(f)
        qs = ",".join("?" for _ in f)
        with self.lock, self.conn:
            self.conn.execute(f"INSERT INTO requests ({cols}) VALUES ({qs})", list(f.values()))

    def spent_today(self) -> float:
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM requests WHERE day=?", (today(),)
            ).fetchone()
        return float(row[0])

    def get_sticky(self, key: str, ttl: int) -> str | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT route FROM sticky WHERE conv_key=? AND ts>?", (key, time.time() - ttl)
            ).fetchone()
        return row[0] if row else None

    def set_sticky(self, key: str, route: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sticky (conv_key, route, ts) VALUES (?,?,?)",
                (key, route, time.time()),
            )

    def stats(self, days: int):
        with self.lock:
            return self.conn.execute(
                """SELECT day, route, COUNT(*), COALESCE(SUM(prompt_tokens),0),
                          COALESCE(SUM(completion_tokens),0), COALESCE(SUM(cost_usd),0),
                          COALESCE(AVG(latency_ms),0)
                   FROM requests WHERE ts > ? GROUP BY day, route ORDER BY day DESC, route""",
                (time.time() - days * 86400,),
            ).fetchall()
