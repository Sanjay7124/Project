import cv2
import numpy as np
import time

def empty(a):
    pass

# Create trackbar window for calibration
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)
cv2.createTrackbar("Pixels per cm", "Parameters", 30, 100, empty)

def stackImages(scale, imgArray):
    """Function to stack multiple images for display"""
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

def getContours(img, imgContour, pixels_per_cm):
    """Function to find contours and measure objects"""
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            
            # Get the perimeter of the contour
            peri = cv2.arcLength(cnt, True)
            
            # Approximate the polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate dimensions in cm
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            
            # Draw the bounding rectangle
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put dimensions text
            cv2.putText(imgContour, f"{width_cm:.1f}cm x {height_cm:.1f}cm", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break
            
        imgContour = img.copy()
        
        # Blur the image
        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        
        # Convert to grayscale
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        # Get trackbar positions
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        pixels_per_cm = cv2.getTrackbarPos("Pixels per cm", "Parameters")
        
        # Make sure pixels_per_cm is at least 1 to avoid division by zero
        if pixels_per_cm < 1:
            pixels_per_cm = 1
        
        # Find edges using Canny
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        
        # Dilate the edges
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        
        # Find and draw contours
        getContours(imgDil, imgContour, pixels_per_cm)
        
        # Stack the images for display
        imgStack = stackImages(0.8, ([img, imgCanny], [imgDil, imgContour]))
        
        # Show the stacked images
        cv2.imshow("Result", imgStack)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()