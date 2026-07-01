import cv2

print("Version:", cv2.__version__)

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("LBPH Available")
except Exception as e:
    print("Error:", e)
    










    