from ultralytics import YOLO
import sys


model = YOLO("models/yolov8m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    name=sys.argv[1],
    data="data/data.yaml",
    epochs=2,
    device=0,
    plots=True,
)
