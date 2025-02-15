import cv2
from pathlib import Path
from ultralytics import YOLO
import os

#run code to test images and get final results

# Path to your YOLO model and test images folder
crying_model_path = r"C:\Users\alzah\Desktop\runs\detect\train19\weights\best.pt"
og_model_path = r"C:\Users\alzah\Downloads\model.pt"

crying_model = YOLO(crying_model_path)
awake_asleep_model = YOLO(og_model_path)
test_images_path = r"C:\Users\alzah\Desktop\dataset\images\shortcutlol" 


# Create 'result' directory if it doesn't exist
result_folder = os.path.join(test_images_path, 'result')
os.makedirs(result_folder, exist_ok=True)

# Process images in the folder
for img_file in Path(test_images_path).glob('*.*'):  # Iterates over all image files
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        print(f"Processing image: {img_file}") #looks cool in cmd prompt
        img = cv2.imread(str(img_file)) #Read the image
        crying_results = crying_model(img) # Perform object detection
        status = "UNKNOWN" #initialising 

        if "crying" in [crying_model.names[int(box.cls)] for box in crying_results[0].boxes]:
            status = "CRYING"
        #not crying, enter if statement to check the other states using the other model lol
        awake_asleep_results = None
        if status == "UNKNOWN":  # Only check awake/asleep if not crying
            awake_asleep_results = awake_asleep_model(img)
            detected_classes = [awake_asleep_model.names[int(box.cls)] for box in awake_asleep_results[0].boxes]
            if "open" in detected_classes:
                status = "AWAKE"
            elif "closed" in detected_classes:
                status = "ASLEEP"
        
        # Print status in terminal
        print(f"Status: {status}")
        # Annotate image with the detection results
        annotated_frame = crying_results[0].plot()
        if awake_asleep_results:
            annotated_frame = awake_asleep_results[0].plot() #overrites the crying one
        cv2.putText(annotated_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
        # Save the annotated image to the result folder
        result_image_path = os.path.join(result_folder, img_file.name)
        cv2.imwrite(result_image_path, annotated_frame)
        print(f"Saved result image to: {result_image_path}")
