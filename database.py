import sqlite3
import time
import random
from datetime import datetime

DB_FILE = "synthnotes.db"

DEFAULT_SECTORS = {
    "IT Services": """Future investments related comments (Including GenAI, AI, Data, Cloud, etc):
Capital allocation:
Talent supply chain related comments:
Org structure change:
Other comments:
Short-term comments:
- Guidance:
- Order booking:
- Impact of macro slowdown:
- Vertical wise comments:""",
    "QSR": """Customer proposition:
Menu strategy (Includes: new product launches, etc):
Operational update (Includes: SSSG, SSTG, Price hike, etc):
Unit economics:
Store opening:"""
}

def _populate_default_sectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sectors")
    if cursor.fetchone()[0] == 0:
        for name, topics in DEFAULT_SECTORS.items():
            cursor.execute("INSERT OR REPLACE INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
        conn.commit()

def init_db():
    with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, meeting_type TEXT,
                file_name TEXT, content TEXT, raw_transcript TEXT,
                refined_transcript TEXT, token_usage INTEGER, processing_time REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sectors ( name TEXT PRIMARY KEY, topics TEXT NOT NULL )
        """)
        conn.commit()
        _populate_default_sectors(conn)

def safe_db_operation(operation_func, *args, **kwargs):
    """Handles 'database is locked' errors with randomized exponential backoff."""
    max_retries = 3
    base_delay = 0.1
    for attempt in range(max_retries):
        try:
            with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
                return operation_func(conn, *args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            else:
                raise ConnectionError(f"Database operation failed after retries: {e}")
    raise ConnectionError("Database remained locked after multiple retries.")


def _save_note_op(conn, note_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notes (id, created_at, meeting_type, file_name, content, raw_transcript, refined_transcript, token_usage, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        note_data.get('id'), note_data.get('created_at'), note_data.get('meeting_type'),
        note_data.get('file_name'), note_data.get('content'), note_data.get('raw_transcript'),
        note_data.get('refined_transcript'), note_data.get('token_usage'), note_data.get('processing_time')
    ))
    conn.commit()

def save_note(note_data: dict):
    safe_db_operation(_save_note_op, note_data)

def _get_all_notes_op(conn, search_query, date_range, meeting_types):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    query = "SELECT * FROM notes WHERE 1=1"
    params = []
    if search_query:
        query += " AND (content LIKE ? OR file_name LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    if date_range and len(date_range) == 2:
        query += " AND date(created_at) BETWEEN ? AND ?"
        params.extend([date_range[0], date_range[1]])
    if meeting_types:
        placeholders = ','.join('?' for _ in meeting_types)
        query += f" AND meeting_type IN ({placeholders})"
        params.extend(meeting_types)
    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]

def get_all_notes(search_query="", date_range=None, meeting_types=None):
    return safe_db_operation(_get_all_notes_op, search_query, date_range, meeting_types)

def get_analytics_summary():
    notes = get_all_notes()
    if not notes:
        return {"total_notes": 0, "avg_time": 0, "total_tokens": 0}
    
    valid_notes_for_avg = [n for n in notes if n.get('processing_time')]
    total_time = sum(n.get('processing_time', 0) for n in valid_notes_for_avg)
    total_tokens = sum(n.get('token_usage', 0) for n in notes if n.get('token_usage'))
    
    return {
        "total_notes": len(notes),
        "avg_time": total_time / len(valid_notes_for_avg) if valid_notes_for_avg else 0,
        "total_tokens": total_tokens
    }

def _get_sectors_op(conn):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name, topics FROM sectors ORDER BY name")
    return {row['name']: row['topics'] for row in cursor.fetchall()}

def get_sectors():
    return safe_db_operation(_get_sectors_op)

def _save_sector_op(conn, name, topics):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
    conn.commit()

def save_sector(name, topics):
    safe_db_operation(_save_sector_op, name, topics)

def _delete_sector_op(conn, name):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sectors WHERE name = ?", (name,))
    conn.commit()

def delete_sector(name):
    safe_db_operation(_delete_sector_op, name)
