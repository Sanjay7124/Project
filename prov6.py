import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Create the Tkinter window
window = tk.Tk()
window.title("Real-Time Object Measurement")
window.geometry("800x600")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up trackbars (parameters tuning)
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)

# Move the 'Parameters' window off-screen
cv2.moveWindow("Parameters", -1000, -1000)

cv2.createTrackbar("Threshold1", "Parameters", 50, 255, lambda a: None)
cv2.createTrackbar("Threshold2", "Parameters", 150, 255, lambda a: None)
cv2.createTrackbar("Area", "Parameters", 3000, 30000, lambda a: None)
cv2.createTrackbar("Pixels per cm", "Parameters", 30, 100, lambda a: None)

# Function to stack multiple images
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    
    return ver

# Function to find contours and measure objects
def getContours(img, imgContour, pixels_per_cm):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            # Use aspect ratio to check if the object is roughly rectangular
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.5 and aspect_ratio < 2:
                width_cm = w / pixels_per_cm
                height_cm = h / pixels_per_cm
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imgContour, f"{width_cm:.1f}cm x {height_cm:.1f}cm", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Function for background processing and updating images
def process_image():
    global imgContour
    success, img = cap.read()  # Capture an image once here
    if not success:
        print("Failed to capture frame from webcam")
        return
    
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    pixels_per_cm = cv2.getTrackbarPos("Pixels per cm", "Parameters")
    
    if pixels_per_cm < 1:
        pixels_per_cm = 1
    
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    
    getContours(imgDil, imgContour, pixels_per_cm)
    
    imgStack = stackImages(0.8, ([img, imgCanny], [imgDil, imgContour]))
    
    # Convert image to RGB and then to Tkinter format
    imgRGB = cv2.cvtColor(imgStack, cv2.COLOR_BGR2RGB)
    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imgRGB))
    
    # Update Tkinter label with the new image
    label.imgTk = imgTk
    label.config(image=imgTk)

    # Schedule the next frame update after 33ms (about 30 FPS)
    window.after(33, process_image)

# Create a Tkinter label to display the webcam feed
label = ttk.Label(window)
label.pack(padx=10, pady=10)

# Function to exit the program when 'q' is pressed
def exit_program(event):
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

# Bind the 'q' key to exit the program
window.bind('<q>', exit_program)

# Start the background image processing loop
window.after(0, process_image)

# Run the Tkinter loop
window.mainloop()

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
