import cv2
from ultralytics import YOLO
import requests

#live detection of the 3 classes

# Load both YOLO models
crying_model_path = r"C:\Users\alzah\Desktop\runs\detect\train19\weights\best.pt" #المودل الجديد
og_model_path = r"C:\Users\alzah\Downloads\model.pt" #الاصلي

crying_model = YOLO(crying_model_path)
awake_asleep_model = YOLO(og_model_path)
 
cap = cv2.VideoCapture(0) # Open webcam

TELEGRAM_BOT_TOKEN = "7674676539:AAFpNqzfwyT7SSacr-g8MB4bk9VfmHhTT8A"
CHAT_ID = "856742481"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    return response.json()

previous_status = "UNKNOWN"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform detection using the crying model first
    crying_results = crying_model(frame)
    detected_classes = [crying_model.names[int(box.cls)] for box in crying_results[0].boxes]
    
    status = "UNKNOWN"
    
    if "crying" in detected_classes:
        status = "!!CRYING!!"
    else:
        # If not crying, check for awake/asleep using the second model
        awake_asleep_results = awake_asleep_model(frame)
        detected_classes = [awake_asleep_model.names[int(box.cls)] for box in awake_asleep_results[0].boxes]
        
        if "open" in detected_classes:
            status = "awake.. uh oh"
        elif "closed" in detected_classes:
            status = "sleeping"

    print(f"Status: {status}")  # Print status in terminal for debugging

    # Send Telegram alert only when status changes
    if status != previous_status and status != "UNKNOWN":
        send_telegram_message(f"Alert: Baby is {status}!")
        previous_status = status  # Update last known status

    # Choose the correct results to plot on the frame
    annotated_frame = crying_results[0].plot() if status == "CRYING" else awake_asleep_results[0].plot()
    
    # i nono like cv2.rectangle(annotated_frame, (40, 30), (350, 100), (255, 255, 255), -1)  # white box
    
    cv2.putText(annotated_frame, f"Baby is {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (75,54,95), 2)  # pinkkk text


    # Show the frame
    cv2.imshow("Live Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


