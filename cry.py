from ultralytics import YOLO
#training of thirds state

# Path to the pretrained model
model_path = r"C:\Users\alzah\Downloads\model.pt" #ofc change path

# Load the pretrained model
model = YOLO(model_path)

# Train the model with the 'crying' data
model.train(data='data.yaml', epochs=10, imgsz=640, device='cpu') 

