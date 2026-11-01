"""Tiny database layer: SQLite by default, Postgres when DATABASE_URL is set.

Render's free web services have an ephemeral filesystem, so the SQLite file is
wiped on every redeploy. Point DATABASE_URL at any Postgres instance (Render,
Neon, Supabase, ...) to make accounts and saved work persistent.
"""
import os
import sqlite3
import threading

DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = 'postgresql://' + DATABASE_URL[len('postgres://'):]

IS_POSTGRES = DATABASE_URL.startswith('postgresql://')
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smileberta.db')

if IS_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        import psycopg2.pool
    except ImportError:
        print('[db] psycopg2 not installed, falling back to SQLite')
        IS_POSTGRES = False

_pool = None
_pool_lock = threading.Lock()


def _pg_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, DATABASE_URL)
    return _pool


class _Conn:
    """Uniform wrapper so app code can use `?` placeholders and dict rows on both engines."""

    def __init__(self):
        if IS_POSTGRES:
            self._raw = _pg_pool().getconn()
        else:
            self._raw = sqlite3.connect(SQLITE_PATH, timeout=15)
            self._raw.row_factory = sqlite3.Row
            self._raw.execute('PRAGMA foreign_keys = ON')

    def _sql(self, sql):
        return sql.replace('?', '%s') if IS_POSTGRES else sql

    def execute(self, sql, params=()):
        if IS_POSTGRES:
            cur = self._raw.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(self._sql(sql), params)
            return cur
        return self._raw.execute(sql, params)

    def insert(self, sql, params=()):
        """Run an INSERT and return the new row id."""
        if IS_POSTGRES:
            cur = self.execute(sql + ' RETURNING id', params)
            return cur.fetchone()['id']
        return self._raw.execute(sql, params).lastrowid

    def one(self, sql, params=()):
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql, params=()):
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def commit(self):
        self._raw.commit()

    def close(self):
        if IS_POSTGRES:
            try:
                self._raw.rollback()
            except Exception:
                pass
            _pg_pool().putconn(self._raw)
        else:
            self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        self.close()


def connect():
    return _Conn()


def is_integrity_error(exc):
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    return IS_POSTGRES and type(exc).__name__ == 'IntegrityError'


_SCHEMA_SQLITE = '''
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fragment_library (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    smiles     TEXT NOT NULL,
    img_data   TEXT,
    props      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    summary    TEXT,
    data       TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_items_user ON items(user_id, kind);
'''

_SCHEMA_PG = '''
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fragment_library (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    smiles     TEXT NOT NULL,
    img_data   TEXT,
    props      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS items (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    summary    TEXT,
    data       TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_items_user ON items(user_id, kind);
'''


def init_db():
    global IS_POSTGRES
    if IS_POSTGRES:
        try:
            connect().close()
        except Exception as e:  # never let a bad DATABASE_URL take the whole API down
            print('[db] Postgres unavailable, falling back to SQLite:', str(e).split('\n')[0])
            IS_POSTGRES = False
    conn = connect()
    try:
        if IS_POSTGRES:
            for stmt in _SCHEMA_PG.split(';'):
                if stmt.strip():
                    conn.execute(stmt)
        else:
            conn._raw.executescript(_SCHEMA_SQLITE)
            # Older SQLite databases had a tokens table without created_at.
            cols = [r['name'] for r in conn.all('PRAGMA table_info(tokens)')]
            if 'created_at' not in cols:
                conn.execute('ALTER TABLE tokens ADD COLUMN created_at TIMESTAMP')
                conn.execute('UPDATE tokens SET created_at = CURRENT_TIMESTAMP')
        conn.commit()
    finally:
        conn.close()
