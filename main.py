# pip install opencv-python-headless
# pip install torch
# pip install torchvision
# pip install ultralytics



# import cv2
# import torch

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Get frame dimensions
#     frame_height, frame_width = frame.shape[:2]

#     # Draw x and y axes
#     cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 255, 0), 1)
#     cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 1)

#     # Perform object detection
#     results = model(frame)
    
#     # Initialize variables to store the highest confidence detection
#     highest_conf = 0
#     best_det = None

#     # Parse results to find the best detection
#     for det in results.xyxy[0]:
#         # Only process 'person' class
#         if det[-1] == 0:  # class 0 in COCO dataset corresponds to 'person'
#             x1, y1, x2, y2, conf, cls = det

#             if conf > highest_conf:
#                 highest_conf = conf
#                 best_det = det

#     # If a detection is found, draw the circle and coordinates
#     if best_det is not None:
#         x1, y1, x2, y2, conf, cls = best_det

#         # Calculate center of the bounding box
#         center_x = int((x1 + x2) / 2)
#         center_y = int((y1 + y2) / 2)
#         radius = int(max(x2 - x1, y2 - y1) / 2)

#         # Draw a red circle with a red dot in the middle
#         cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
#         cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

#         # Display coordinates
#         coordinates_text = f"(x:{center_x}, y:{center_y})"
#         cv2.putText(frame, coordinates_text, (center_x + 10, center_y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#     # Display the frame with detections and axes
#     cv2.imshow('AI Target Tracking', frame)

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()





import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    
    # Parse results
    for det in results.xyxy[0]:
        # Only process 'person' class
        if det[-1] == 0:  # class 0 in COCO dataset corresponds to 'person'
            x1, y1, x2, y2, conf, cls = det

            # Calculate center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)

            # Draw a red circle with a red dot in the middle
            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Display the frame with detections
    cv2.imshow('AI Target Tracking', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
