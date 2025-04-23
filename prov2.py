import cv2
import numpy as np

# Known width of the reference object (in cm)
REFERENCE_WIDTH_CM = 8.56  # Width of credit card

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 100)
    dilated = cv2.dilate(edged, None, iterations=1)
    return cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def measure_object(frame, ref_width_cm=REFERENCE_WIDTH_CM):
    contours = find_contours(frame)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    pixels_per_cm = None

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = sorted(box, key=lambda x: x[0])  # sort for consistent corner ordering
        box = np.array(box)  # convert back to np.array after sorting

        # Reshape for drawContours
        cv2.drawContours(frame, [box.reshape((-1, 1, 2))], -1, (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        width = np.linalg.norm([tltrX - blbrX, tltrY - blbrY])
        height = np.linalg.norm([tlblX - trbrX, tlblY - trbrY])

        if pixels_per_cm is None:
            pixels_per_cm = width / ref_width_cm
            continue  # skip reference itself

        object_width = width / pixels_per_cm
        object_height = height / pixels_per_cm

        # Draw size labels on the object
        cv2.putText(frame, f"{object_width:.1f} cm", (int(tltrX), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"{object_height:.1f} cm", (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    measured_frame = measure_object(frame.copy())
    cv2.imshow("Object Measurement", measured_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
