import os
import sqlite3

os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/face_db.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    dataset_path TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT,
    date TEXT,
    time TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

cursor.execute("PRAGMA table_info(attendance)")
attendance_columns = [row[1] for row in cursor.fetchall()]
if "user_id" not in attendance_columns:
    cursor.execute("ALTER TABLE attendance ADD COLUMN user_id INTEGER")

conn.commit()
conn.close()

print("Database Created")