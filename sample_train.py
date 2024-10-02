from ultralytics import YOLO

# Load a model
model = YOLO("yolov10x.yaml")
model = YOLO("yolov10x.pt")
model = YOLO("yolov10x.yaml").load("yolov10x.pt")

# Train the model
results = model.train(data="minedata.yaml", epochs=100, imgsz=640, 
                      project="mining_object_detection", name="yolov10")