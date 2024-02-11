from ultralytics import YOLO
import sys


model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    name=sys.argv[1],
    data="data/data.yaml",
    epochs=5,
    device=0,
    plots=True,
    optimizer=sys.argv[2],
)
