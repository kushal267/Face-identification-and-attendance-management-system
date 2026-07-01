import sqlite3

conn = sqlite3.connect("database/face_db.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM attendance")

rows = cursor.fetchall()

print(rows)

cursor.execute("SELECT COUNT(*) FROM attendance")

print(cursor.fetchone())

conn.close()
