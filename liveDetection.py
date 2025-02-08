import cv2
from ultralytics import YOLO

# Load YOLO model
model_path = r"C:\Users\alzah\Downloads\model.pt"  
model = YOLO(model_path)

# Open webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform object detection
    results = model(frame)

    # Initialize status
    status = "UNKNOWN"

    # Extract detected class names
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]

    if "open" in detected_classes:
        status = "AWAKE"
    elif "closed" in detected_classes:
        status = "ASLEEP"

    # Print status in terminal
        print(f"Status: {status}")
        
        # Print status in terminal
        print(f"Status: {status}")

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Display status text on the frame
    cv2.putText(annotated_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Live Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
