import sqlite3
from datetime import datetime

DB_FILE = "synthnotes.db"

# --- Default Data for First-Time Setup ---
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
    """Internal function to add default sectors if the table is empty."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sectors")
    if cursor.fetchone()[0] == 0:
        for name, topics in DEFAULT_SECTORS.items():
            cursor.execute("INSERT INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
        conn.commit()

def init_db():
    """Initializes the database and creates/updates tables."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, meeting_type TEXT,
                file_name TEXT, content TEXT, raw_transcript TEXT,
                refined_transcript TEXT, token_usage INTEGER, processing_time REAL
            )
        """)
        # NEW: Sectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sectors (
                name TEXT PRIMARY KEY,
                topics TEXT NOT NULL
            )
        """)
        conn.commit()
        # Populate with defaults if needed
        _populate_default_sectors(conn)

def save_note(note_data: dict):
    """Saves a generated note to the database."""
    with sqlite3.connect(DB_FILE) as conn:
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

def get_all_notes(search_query="", date_range=None, meeting_types=None):
    """Retrieves all notes, with optional filtering."""
    with sqlite3.connect(DB_FILE) as conn:
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

def get_analytics_summary():
    """Calculates summary metrics from the database."""
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

# --- NEW FUNCTIONS FOR SECTOR MANAGEMENT ---

def get_sectors() -> dict:
    """Retrieves all custom sectors and their topics from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT name, topics FROM sectors ORDER BY name")
        return {row['name']: row['topics'] for row in cursor.fetchall()}

def save_sector(name: str, topics: str):
    """Saves (inserts or replaces) a sector template."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
        conn.commit()

def delete_sector(name: str):
    """Deletes a sector template."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sectors WHERE name = ?", (name,))
        conn.commit()
