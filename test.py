""" # Copy code
import ultralytics
import cv2

This is a code for testing yolov8n on a webcam.

model = ultralytics.YOLO('yolov8n.pt')
# Open a live webcam feed
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		break
	results = model.predict(frame)
	results[0].show()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
#############################################################################################################
What is the purpose of this code snippet below?
 Absolutely! Let's break down the logic behind calculating the distance from the camera to an object using the known parameters and the object detection bounding box. The process involves basic principles of geometry and optics.

Key Concepts
Camera Calibration:

The focal length of the camera lens is crucial. It determines how strongly the camera converges or diverges light. The focal length is typically measured in pixels, and it can be found using camera calibration techniques.
Real-World Height:

This is the actual height of the object you want to measure the distance to, expressed in meters. For example, if you are measuring a person who is approximately 1.7 meters tall, you would use this value.
Object Height in the Image:

When the object is detected in the video frame, the model provides bounding box coordinates. The height of this bounding box in pixels represents how tall the object appears in the image.
Distance Calculation Formula
The distance to the object can be calculated using the following formula derived from similar triangles:

Distance =
(Actual Height) * (Focal Length) / (Object Height in Image) 

"""

import cv2
import ultralytics

# Load the model
model = ultralytics.YOLO('yolov8n.pt')

# Known parameters
ACTUAL_HEIGHT = 1.7  # Real-world height in meters
FOCAL_LENGTH = 800  # Camera focal length in pixels (estimate this via calibration)

def calculate_distance(actual_height, focal_length, object_height_in_image):
    return (actual_height * focal_length) / object_height_in_image

# Open a live webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects in the frame
    results = model.predict(frame)
    print(results)  # Debugging line

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates, convert to integers
            object_height_in_image = y2 - y1  # Height of bounding box in pixels
            
            # Calculate distance
            distance = calculate_distance(ACTUAL_HEIGHT, FOCAL_LENGTH, object_height_in_image)
            
            # Draw bounding box and distance text on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, f"{distance:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)  # Distance text above the box

    # Display the frame with bounding boxes and distance
    cv2.imshow("Object Distance", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
