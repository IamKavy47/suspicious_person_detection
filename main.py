import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load familiar faces for DeepFace
db_path = "/workspaces/suspicious_person_detection/face_db/"  # Path to your database of familiar faces

# Access the phone's IP camera stream
stream_url = "http://192.168.29.74:4747/video"  # Replace with the IP camera stream URL from your phone
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO human detection
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        if int(box.cls) == 0:  # Class 0 = "person"
            # Get the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the face region
            face_image = frame[y1:y2, x1:x2]
            
            # Perform face recognition
            try:
                result = DeepFace.find(img_path=face_image, db_path=db_path, enforce_detection=False)
                if len(result) > 0 and not result[0].empty:
                    label = "Familiar"
                    color = (0, 255, 0)  # Green for familiar
                else:
                    label = "Suspicious"
                    color = (0, 0, 255)  # Red for suspicious
            except:
                label = "No Match"
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the live feed
    cv2.imshow("Live Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
