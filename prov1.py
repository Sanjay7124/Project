import cv2
import numpy as np

# Reference object width in cm (e.g., a known card or object)
KNOWN_WIDTH_CM = 8.56  
EXPECTED_PIXEL_WIDTH = 200  # Approximate width in pixels when the object is 20 cm away

def get_contours(frame):
    """ Detect contours in the frame """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_pixel_per_cm(contours, frame):
    """ Find object size and ensure it's within 20 cm """
    selected_contour = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if object width is within the expected pixel range for 20 cm
        if abs(w - EXPECTED_PIXEL_WIDTH) < 20:  # Allow small error margin
            selected_contour = cnt
            return w / KNOWN_WIDTH_CM, selected_contour  # Pixel-to-cm ratio

    return None, None

def measure_object(frame, contour, pixel_per_cm):
    """ Measure and display object dimensions """
    if contour is None or pixel_per_cm is None:
        return

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = max(rect[1])
    height = min(rect[1])

    width_cm = width / pixel_per_cm
    height_cm = height / pixel_per_cm

    # Draw rectangle and display measurement
    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    cv2.putText(frame, f"{width_cm:.1f}cm x {height_cm:.1f}cm",
                (int(rect[0][0] - 40), int(rect[0][1] - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        contours = get_contours(frame)
        pixel_per_cm, selected_contour = get_pixel_per_cm(contours, frame)

        if pixel_per_cm:
            measure_object(frame, selected_contour, pixel_per_cm)

        cv2.imshow("Object Measurement (20cm Only)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
