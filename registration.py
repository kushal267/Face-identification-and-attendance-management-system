import cv2
import os
import sqlite3

name = input("Enter Name: ").strip()
if not name:
    print("Name cannot be empty.")
    exit(1)

safe_name = name.replace(" ", "_")

os.makedirs("dataset", exist_ok=True)
os.makedirs("database", exist_ok=True)

db_path = os.path.join("database", "face_db.db")
conn = sqlite3.connect(db_path)
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

cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
user_id = cursor.lastrowid
folder_name = f"{user_id}_{safe_name}"
path = os.path.join("dataset", folder_name)

cursor.execute(
    "UPDATE users SET dataset_path = ? WHERE user_id = ?",
    (path, user_id)
)
conn.commit()
conn.close()

if not os.path.exists(path):
    os.makedirs(path)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 
    "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open webcam. Check camera connection.")
    exit(1)

count = 0

while True:

    ret, img = cam.read()
    if not ret or img is None:
        print("Warning: Failed to read a frame from webcam")
        break

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x,y,w,h) in faces:

        count += 1

        cv2.imwrite(
            f"{path}/{count}.jpg",
            gray[y:y+h,x:x+w]
        )

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

    cv2.imshow("Register Face", img)

    if cv2.waitKey(100) & 0xff == 27:
        break

    elif count >= 50:
        break

cam.release()
cv2.destroyAllWindows()

print("Face Registered")