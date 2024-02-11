# MLflow learning repo

This repository demonstrates the use of MLflow for managing machine learning experiments and tracking results, with a focus on object detection using the Ultralytics YOLO model.

## Prerequisites

Ensure you have Python installed on your machine. This project is built and tested with Python 3.11.

```
pip install mlflow 
pip install ultralytics
```
## Running MLflow

Start the MLflow tracking server to manage experiment metadata and results. This setup allows the Ultralytics models to integrate with the MLflow backend, facilitating the tracking of training processes and outcomes.

Execute the following command in your terminal to initiate the MLflow server:

```
mlflow server --backend-store-uri runs/mlflow
```

This command starts the MLflow tracking UI, which is accessible by default at http://127.0.0.1:5000.

# Data
The [data](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection?resource=download) used in this example is a brain tumor detection data set. Each image is labled with the region connected to a brain tumor. 

## Ultralytics YOLO Model
Ultralytics offers a range of models for different applications. For this project, we employ the YOLOv8n model from Ultralytics for object detection. YOLO (You Only Look Once) is a cutting-edge model known for its efficiency and accuracy in detecting objects within images.

The choice of YOLOv8n, the smallest variant, was made to minimize processing time for this demonstration. However, it's important to note that for actual applications requiring higher accuracy or processing larger images, a more robust version of the YOLO model would be preferable.

## Experiments
The primary focus of these experiments is to evaluate the effectiveness of different optimizers for brain tumor detection, specifically comparing SGD, Adam, and Adamax. Each optimizer was tested across five epochs, with the results available for review in the MLflow web application.

Although only a limited number of epochs were run, making the results inconclusive, these experiments showcase the potential of using MLflow for real-time model training monitoring, including comparing optimizer performance, inspecting metrics, and reviewing result artifacts.



