# Face Recognition Attendance System

A simple face recognition attendance system built with Python, OpenCV, SQLite, and Tkinter.

## Features

- Register new students with live camera face capture
- Train a DeepFace embedding model from registered face images
- Detect faces through webcam and mark attendance automatically
- Prevent duplicate attendance entries for the same student on the same day
- View attendance records in a table
- Export attendance data to Excel (`attendance/attendance.xlsx`)

## Repository Structure

- `gui.py` - main Tkinter user interface
- `registration.py` - capture face images for a student
- `training.py` - train the DeepFace model and save `face_model.pkl`
- `recognition.py` - run live face recognition and mark attendance
- `export_excel.py` - export the attendance table to Excel
- `database.py` - create SQLite database and attendance table
- `attendance/` - exported Excel reports
- `dataset/` - face images organized by student name
- `database/face_db.db` - SQLite database file
- `face_model.pkl` - trained DeepFace model
- `haarcascade_frontalface_default.xml` - Haar cascade face detector

## Dependencies

Install the required Python packages before using the project.

```bash
pip install opencv-python pandas deepface tf-keras
```

If you plan to run the GUI, make sure Tkinter is installed and available in your Python distribution.

## Setup

1. Clone or copy this project to your local machine.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python pandas deepface tf-keras
   ```
4. Ensure the `attendance/` and `database/` folders exist.
5. Run `database.py` once to create the SQLite database and attendance table.

## How to Use

### 1. Register Face
- Open `gui.py` or run the app from terminal:
  ```bash
  python gui.py
  ```
- Click `Register Face`.
- Enter the student name in the terminal.
- The camera will capture face images.
- Press `Esc` after enough samples are recorded.

### 2. Train Model
- Click `Train Model` in the GUI.
- This reads all face images from `dataset/`, trains a KNN model, and saves `face_model.pkl`.

### 3. Start Identification
- Click `Start Identification` to open webcam recognition.
- The system will detect faces and mark attendance automatically.
- Attendance records are stored in `database/face_db.db`.

### 4. View Attendance
- Click `View Attendance` to open the attendance records table.

### 5. Export Excel
- Click `Export Excel` to generate the report.
- The Excel file is saved here:
  ```text
  attendance/attendance.xlsx
  ```

### 6. Reset Attendance
- Use the `Reset Attendance` button in the GUI to clear all attendance records from the database.
- This action deletes only the attendance entries, not the registered face dataset or trained model.

## Notes

- `attendance.xlsx` is a binary Excel workbook file; open it with Excel, LibreOffice Calc, or Google Sheets.
- If the file is not visible in a text editor, that is normal because `.xlsx` is not plain text.

## Recommended Improvements

- Add a GUI-based student management panel for editing and deleting records
- Add date filters and summary statistics in the attendance view
- Add exception handling for missing camera or missing model file
- Add CSV export in addition to Excel
- Package the app with PyInstaller for distribution

## Troubleshooting

- If face recognition fails, make sure `face_model.pkl` exists and `dataset/` contains labeled face images.
- If DeepFace import or model loading fails, install `deepface` and `tf-keras`.
- If camera does not open, check that your webcam is connected and available.
- If Excel export fails, verify that `pandas` is installed and `attendance/` folder exists.

## License

This project is provided as-is for learning and personal use.
