import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Create the Tkinter window
window = tk.Tk()
window.title("Real-Time Object Measurement")
window.geometry("900x700")

# Initialize webcam with fixed resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters (adjustable)
threshold1 = 50
threshold2 = 150
min_area = 5000
pixels_per_cm = 40  # Adjust by testing with a known object

# Function to find contours and measure objects
def getContours(img, imgContour, pixels_per_cm):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            # Filter out long shapes or very small/large ratios
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:
                width_cm = w / pixels_per_cm
                height_cm = h / pixels_per_cm

                # Draw only if the object is reasonably centered
                img_center_x = imgContour.shape[1] // 2
                obj_center_x = x + w // 2
                if abs(obj_center_x - img_center_x) < imgContour.shape[1] // 3:
                    cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(imgContour, f"{width_cm:.1f}cm x {height_cm:.1f}cm",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    detected = True

    if not detected:
        cv2.putText(imgContour, "No valid object detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Process and update webcam frames
def process_image():
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam")
        return

    img = cv2.resize(img, (900, 600))  # Resize to fit window
    imgContour = img.copy()

    imgGray = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 1), cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    imgDil = cv2.dilate(imgCanny, np.ones((5, 5), np.uint8), iterations=1)

    getContours(imgDil, imgContour, pixels_per_cm)

    # Convert to Tkinter format
    imgRGB = cv2.cvtColor(imgContour, cv2.COLOR_BGR2RGB)
    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imgRGB))
    label.imgTk = imgTk
    label.config(image=imgTk)

    window.after(33, process_image)

# Tkinter Label for webcam display
label = ttk.Label(window)
label.pack(padx=10, pady=10, fill="both", expand=True)

# Quit app on 'q'
def exit_program(event):
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

window.bind('<q>', exit_program)

# Start webcam feed loop
window.after(0, process_image)
window.mainloop()

# Cleanup after close
cap.release()
cv2.destroyAllWindows()
