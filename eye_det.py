import cv2
import os
import time

# Load the eye classifier
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Create a directory to save captured images
output_dir = "captured_eyes"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to detect eyes and draw bounding boxes
def detect_bounding_box(vid):
    # Convert the video frame to grayscale
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the grayscale image
    eyes = eye_classifier.detectMultiScale(gray_image, 1.1, 10, minSize=(20, 20))
    # Iterate over the detected eyes
    for (x, y, w, h) in eyes:
        # Draw a rectangle around each eye
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return eyes

last_saved_time = time.time()
capture_interval = 2  # Interval in seconds

while True:
    # Read a frame from the video capture object
    result, video_frame = video_capture.read()
    # If the frame was not read successfully, break the loop
    if not result:
        break
    
    # Detect eyes in the video frame
    eyes = detect_bounding_box(video_frame)
    
    # Display the video frame with detected eyes
    cv2.imshow("Eye Detection Project", video_frame)
    
    # Capture and save the image every 2 seconds
    current_time = time.time()
    if current_time - last_saved_time >= capture_interval:
        # Save the captured image with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f"{output_dir}/eye_{timestamp}.jpg"
        cv2.imwrite(image_path, video_frame)
        last_saved_time = current_time
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
