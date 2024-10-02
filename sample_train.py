from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.yaml")
model = YOLO("yolov8x.pt")
model = YOLO("yolov8x.yaml").load("yolov8x.pt")

# Train the model
results = model.train(data="minedata.yaml", epochs=100, imgsz=640, 
                      project="mining_object_detection", name="experiement")