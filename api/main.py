from collections import defaultdict
from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from keras.models import load_model
import tempfile
from ultralytics import YOLO
from db import create_connection
from save_db import create_plate_detection_table, insert_detection, load_region_mapping_from_csv

from util import read_license_plate, get_license_plate_region

app = FastAPI()
model = load_model("models/ssd_base.h5")
yolo_model = YOLO("models/best.pt")
load_region_mapping_from_csv("data/Book 7.csv")



def detect_objects_on_frame(frame):
    prediction = model.predict(frame)
    prediction = prediction[0] * 300
    prediction = np.array(prediction, dtype=np.uint8)
    return prediction


def preprocess_frame(image, target_size=(300, 300)):
    img = cv2.resize(image, target_size)
    img_array = np.array(img) / 255
    img_array = img_array.reshape(1, *target_size, 3)
    return img_array


def yolo_detection(frame):
    yolo_predicted = yolo_model(frame)[0]
    results = defaultdict(dict)
    for license_plate in yolo_predicted.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop license plate
        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

        # Process license plate
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop, cv2.COLOR_BGR2GRAY
        )
        # Adjusting the threshold value
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY
            )


        # Read license plate number
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )

        if license_plate_text is not None:
            results[class_id] = {
                "license_plate": {
                    "bounding_box": [x1, y1, x2, y2],
                    "text": license_plate_text,
                    "bbox_score": score,
                    "text_score": license_plate_text_score,
                    "region": get_license_plate_region(license_plate_text),
                }
            }
    return results


def save_db(detection):
    conn = create_connection()
    create_plate_detection_table(conn)

    license_plate_info = detection.get("license_plate", {})
    bounding_box = license_plate_info.get("bounding_box", [0, 0, 0, 0])
    text = license_plate_info.get("text", "")
    bbox_score = license_plate_info.get("bbox_score", 0)
    text_score = license_plate_info.get("text_score", 0)
    xmin, ymin, xmax, ymax = bounding_box
    dt = [xmin, ymin, xmax, ymax, text, bbox_score, text_score]
    insert_detection(conn, dt)


@app.post("/prediction")
async def prediction(file: UploadFile):
    if file.content_type.startswith("video"):
        video_bytes = await file.read()
        temp_file_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(video_bytes)

        cap = cv2.VideoCapture(temp_file_path)

        results = []

        while True:
            ret, frame = cap.read()
            print(ret)
            if not ret:
                break

            # processed_frame = preprocess_frame(frame)
            # frame_results = detect_objects_on_frame(processed_frame)
            image_results_yolo = yolo_detection(frame)
            results.append(image_results_yolo[0])

        cap.release()

        return {"results": results}
    elif file.content_type.startswith("image"):
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # processed_image = preprocess_frame(image)
        # image_results = detect_objects_on_frame(processed_image)
        image_results_yolo = yolo_detection(image)
        image_results_yolo = image_results_yolo[0]
        save_db(image_results_yolo)
        return image_results_yolo
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a video or an image.",
        )
