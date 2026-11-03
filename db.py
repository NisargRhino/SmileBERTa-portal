"""Tiny database layer: SQLite by default, Postgres when DATABASE_URL is set.

Render's free web services have an ephemeral filesystem, so the SQLite file is
wiped on every redeploy. Point DATABASE_URL at any Postgres instance (Render,
Neon, Supabase, ...) to make accounts and saved work persistent.
"""
import os
import sqlite3
import time

DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = 'postgresql://' + DATABASE_URL[len('postgres://'):]

IS_POSTGRES = DATABASE_URL.startswith('postgresql://')
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smileberta.db')

if IS_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print('[db] psycopg2 not installed, falling back to SQLite')
        IS_POSTGRES = False


class DatabaseUnavailable(RuntimeError):
    """Raised when the database cannot be reached (mapped to HTTP 503 by the app)."""


def _pg_connect():
    """A fresh connection per request.

    Hosted Postgres (Neon, Supabase, Render) closes idle connections and may
    suspend when idle; pooled connections then fail at random. A new connection
    costs ~50 ms and is always healthy. One retry covers a cold-starting database.
    """
    last = None
    for attempt in range(3):
        try:
            return psycopg2.connect(DATABASE_URL, connect_timeout=15, keepalives=1,
                                    keepalives_idle=30, keepalives_interval=10, keepalives_count=3)
        except psycopg2.OperationalError as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise DatabaseUnavailable(str(last).split('\n')[0])


class _Conn:
    """Uniform wrapper so app code can use `?` placeholders and dict rows on both engines."""

    def __init__(self):
        if IS_POSTGRES:
            self._raw = _pg_connect()
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
        try:
            if IS_POSTGRES:
                self._raw.rollback()
        except Exception:
            pass
        try:
            self._raw.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        self.close()


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
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
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
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_lower ON users (lower(username));
'''


_schema_ready = False
_schema_lock = __import__('threading').Lock()


def _create_schema():
    conn = _Conn()
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
            try:
                conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_lower ON users (lower(username))')
            except sqlite3.IntegrityError:
                pass  # legacy local DB with case-duplicate usernames
        conn.commit()
    finally:
        conn.close()


def ensure_schema():
    """Create tables once. Retried on later requests if the database was unreachable at boot.

    When DATABASE_URL is set we never fall back to SQLite: silently switching to an
    ephemeral file is exactly what made accounts disappear.
    """
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if not _schema_ready:
            _create_schema()
            _schema_ready = True


def connect():
    ensure_schema()
    return _Conn()


def init_db():
    try:
        ensure_schema()
    except DatabaseUnavailable as e:
        print('[db] Database not reachable yet, will retry on first request:', e)
