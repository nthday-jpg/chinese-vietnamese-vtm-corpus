"""Small transactional page queue shared by isolated OCR processes (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager


class GPUPageQueue:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA busy_timeout=30000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self, jobs, totals):
        """Called once per notebook attempt, using only pages without checkpoints."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE jobs (id INTEGER PRIMARY KEY, engine TEXT, book TEXT, '
                       'page INTEGER, payload TEXT, state TEXT, owner TEXT, seconds REAL, '
                       'UNIQUE(engine,book,page))')
            db.execute('CREATE INDEX claim_job ON jobs(engine,book,state,id)')
            db.execute('CREATE TABLE models (engine TEXT PRIMARY KEY, total INTEGER, '
                       'cached INTEGER, load_seconds REAL DEFAULT 0)')
            for engine, (total, cached) in totals.items():
                db.execute('INSERT INTO models(engine,total,cached) VALUES (?,?,?)', (engine,total,cached))
            db.executemany('INSERT INTO jobs(engine,book,page,payload,state) VALUES (?,?,?,?,?)',
                ((engine, book, task['page'], json.dumps(task), 'pending') for engine, book, task in jobs))

    def claim(self, engine, book, owner):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id,payload FROM jobs WHERE engine=? AND book=? AND "
                             "(state='pending' OR (state='reserved' AND owner=?)) "
                             "ORDER BY CASE state WHEN 'reserved' THEN 0 ELSE 1 END,id LIMIT 1",
                             (engine,book,owner)).fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running',owner=? WHERE id=?", (owner,row[0]))
            db.commit()
            return json.loads(row[1]) if row else None

    def reserve(self, engine, owner):
        """Keep one page for an incoming helper while it imports/loads the model."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id FROM jobs WHERE engine=? AND state='pending' ORDER BY id LIMIT 1",
                             (engine,)).fetchone()
            if row:
                db.execute("UPDATE jobs SET state='reserved',owner=? WHERE id=?", (owner,row[0]))
            return row is not None

    def pages(self, engine, book, owner):
        while True:
            task = self.claim(engine, book, owner)
            if task is None:
                return
            yield task

    def finish(self, engine, book, page, owner, *, success, seconds=0):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?,seconds=? WHERE engine=? AND book=? AND page=? "
                       "AND owner=? AND state='running'",
                       ('done' if success else 'failed', seconds, engine, book, page, owner))

    def abandon(self, owner):
        # Hard crashes must not strand a lease or loop on a bad page this attempt.
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='failed' WHERE owner=? AND state IN ('running','reserved')", (owner,))

    def loaded(self, engine, seconds):
        with self.connect() as db:
            db.execute('UPDATE models SET load_seconds=? WHERE engine=?', (seconds,engine))

    def snapshot(self):
        with self.connect() as db:
            models = {n: dict(pages=total, done=cached, missing=total-cached,
                             pending=0, running=0, failed=0, load_seconds=load, page_seconds=0)
                      for n,total,cached,load in db.execute('SELECT * FROM models')}
            for engine,state,count,seconds in db.execute(
                    'SELECT engine,state,COUNT(*),AVG(seconds) FROM jobs GROUP BY engine,state'):
                if state == 'done':
                    models[engine]['done'] += count
                    models[engine]['missing'] -= count
                    models[engine]['page_seconds'] = seconds or 0
                else:
                    key = 'running' if state == 'reserved' else state
                    models[engine][key] += count
            return models
