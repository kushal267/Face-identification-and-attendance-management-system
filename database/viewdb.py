import sqlite3

conn = sqlite3.connect("database/face_db.db")
cursor = conn.cursor()

cursor.execute("SELECT name, date, time FROM attendance")
rows = cursor.fetchall()

if not rows:
    print("No attendance records found in database.")
else:
    for row in rows:
        print(row)

conn.close()