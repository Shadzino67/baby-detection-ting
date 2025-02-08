import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# Path to YOLO model and test images folder
model_path = r"C:\Users\alzah\Downloads\model.pt"
test_images_path = r"C:\Users\alzah\Desktop\testimage"  

# Load YOLO model
model = YOLO(model_path)
model.yaml['nc'] = 3  # Update to 3 classe

# Train model using new data (with 3 classes)
model.train(data='data.yaml', epochs=10, batch=20)

# Create 'result' directory if it doesn't exist
result_folder = os.path.join(test_images_path, 'result')
os.makedirs(result_folder, exist_ok=True)


# Process images in the folder
for img_file in Path(test_images_path).glob('*.*'):  # Iterates over all image files
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        print(f"Processing image: {img_file}")
        
        # Read the image
        img = cv2.imread(str(img_file))
        
        # Perform object detection
        results = model(img)
        
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

        # Annotate image with the detection results
        annotated_frame = results[0].plot()

        # Save the annotated image to the result folder
        result_image_path = os.path.join(result_folder, img_file.name)
        cv2.imwrite(result_image_path, annotated_frame)
        print(f"Saved result image to: {result_image_path}")
