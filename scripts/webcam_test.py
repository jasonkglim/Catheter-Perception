import cv2
from datetime import datetime
import subprocess

def configure_camera(devices):
    for device in devices:

        print(f"Configuring camera on {device}...")

        # Build the commands to configure the camera
        commands = [
            f"v4l2-ctl -d {device} -c focus_automatic_continuous=0",
            f"v4l2-ctl -d {device} -c auto_exposure=1",
            # f"v4l2-ctl -d {device} -c white_balance_automatic=0",
            f"v4l2-ctl -d {device} -c focus_absolute=76",
            f"v4l2-ctl -d {device} -c exposure_time_absolute=76",
            # f"v4l2-ctl -d {device} -c white_balance_temperature=4675",
            f"v4l2-ctl -d {device} -c gain=0",
            # f"v4l2-ctl -d {device} -c brightness=128",
            # f"v4l2-ctl -d {device} -c contrast=128",
            # f"v4l2-ctl -d {device} -c saturation=128",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

        print("Camera configuration complete!")

print("Testing ability to open webcam device and stream video...")
device = "/dev/video0"
# configure_camera([device])

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # or "/dev/video0"

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
