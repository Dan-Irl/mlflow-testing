from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # load a custom model

# Define path to directory containing images and videos for inference
source = "data/test/images"  # or file, directory, or glob. e.g.  "data/images/*.jpg

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

i = 0
# Show the results
for r in results:
    i += 1
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(f"test_results/test_{i}.jpg")  # save image
