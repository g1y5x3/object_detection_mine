from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.yaml")  # build a new model from YAML
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8x.yaml").load("yolov8x.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="minedata.yaml", epochs=100, imgsz=640)