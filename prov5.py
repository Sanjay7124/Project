import cv2
import imutils
from imutils.video import VideoStream
import time
import numpy as np

# Define the width of your reference object in centimeters (credit card width)
REFERENCE_WIDTH_CM = 8.56

def find_marker(image):
    # Convert to grayscale and blur slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(gray, 35, 125)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return None

    # Sort by area and grab the largest
    c = max(cnts, key=cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

# Start webcam stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Calibrate focal length using reference object
print("[INFO] Calibrating with reference object...")
while True:
    frame = vs.read()
    marker = find_marker(frame)
    if marker:
        focalLength = (marker[1][0] * 30) / REFERENCE_WIDTH_CM
        print(f"[INFO] Focal Length Estimated: {focalLength}")
        break
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()

print("[INFO] Measuring object size... Press 'q' to quit.")
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    marker = find_marker(frame)

    if marker:
        box = cv2.boxPoints(marker)
        box = box.astype(np.intp)  # Fixed version of np.int0
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        pixel_width = min(marker[1][0], marker[1][1])
        pixel_height = max(marker[1][0], marker[1][1])
        per_cm = pixel_width / REFERENCE_WIDTH_CM

        width_cm = pixel_width / per_cm
        height_cm = pixel_height / per_cm

        cv2.putText(frame, f"W: {width_cm:.1f} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"H: {height_cm:.1f} cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Live Measurement", frame)
    if cv2.waitKey(1) == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
