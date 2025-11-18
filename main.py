from ultralytics import YOLO

model = YOLO("yolo11l.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="mydata.yaml", epochs=30)

