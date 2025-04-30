import cv2
from datetime import datetime

print("Testing ability to open webcam device and stream video...")
cap = cv2.VideoCapture(0)  # or "/dev/video0"

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    cv2.imshow('Webcam Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Quit
        break
    elif key == ord('s'):
        # Save current frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"images/frame_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved current frame to {filename}")

cap.release()
cv2.destroyAllWindows()
