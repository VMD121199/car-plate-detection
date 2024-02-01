from collections import defaultdict
from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from keras.models import load_model
import tempfile
from ultralytics import YOLO
from db import create_connection
from auth import create_users_table, get_user_by_email, insert_user
from save_db import (
    create_plate_detection_table,
    insert_detection,
)

# we use sort extension to track vehicle
# install sort by: git clone https://github.com/abewley/sort.git
# from sort.sort import *

from util import (
    # get_car,
    read_license_plate,
    get_license_plate_region,
    load_region_mapping_from_csv,
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()
# model = load_model("../model/ssd_base.h5")
yolo_model = YOLO("../model/best.pt")
coco_model = YOLO("../model/yolov8n.pt")
load_region_mapping_from_csv("../data/Book 7.csv")
# mot_tracker = Sort()
vehicles = [2, 3, 5, 7]

conn = create_connection()
create_plate_detection_table(conn)
create_users_table(conn)


def detect_objects_on_frame(frame):
    prediction = yolo_model.predict(frame)
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
    detections = coco_model(frame)[0]
    detections_ = []
    results = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track_ids = mot_tracker.update(np.asarray(detections_))

    for license_plate in yolo_predicted.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        # if car_id != -1:
        # Crop license plate
        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop, cv2.COLOR_BGR2GRAY
        )
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 125, 255, cv2.THRESH_BINARY
        )
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )
        if (
            license_plate_text is not None
            and license_plate_text_score >= 0.55
            and score >= 0.55
        ):
            results.append(
                {
                    # "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bounding_box": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": score,
                        "text_score": license_plate_text_score,
                        "region": get_license_plate_region(license_plate_text),
                    },
                }
            )
    return results


def save_db(detection):
    license_plate_info = detection.get("license_plate", {})
    bounding_box = license_plate_info.get("bounding_box", [0, 0, 0, 0])
    text = license_plate_info.get("text", "")
    bbox_score = license_plate_info.get("bbox_score", 0)
    text_score = license_plate_info.get("text_score", 0)
    region = license_plate_info.get("region", "")
    xmin, ymin, xmax, ymax = bounding_box
    dt = [xmin, ymin, xmax, ymax, text, bbox_score, text_score, region]
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
        save_count = 15
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # processed_frame = preprocess_frame(frame)
            # frame_results = detect_objects_on_frame(processed_frame)
            image_results_yolo = yolo_detection(frame)
            results.append(image_results_yolo)
            print(image_results_yolo)
            if save_count == 15:
                for detected in image_results_yolo:
                    save_db(detected)
                save_count = 0
            save_count += 1

        cap.release()
        return {"results": results}
    elif file.content_type.startswith("image"):
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # processed_image = preprocess_frame(image)
        # image_results = detect_objects_on_frame(processed_image)
        image_results_yolo = yolo_detection(image)
        for detected in image_results_yolo:
            save_db(detected)
        return image_results_yolo
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a video or an image.",
        )


@app.post("/login")
async def login(data: dict):
    email = data["email"]
    password = data["password"]
    user = get_user_by_email(conn, email)
    if user is not None:
        stored_password = user[2]
        if password == stored_password:
            return {"login": True, "msg": f"Logged in as {email}"}
        else:
            return {"login": False, "msg": "Incorrect password"}
    else:
        return {"login": False, "msg": "User not exists"}


@app.post("/signup")
async def signup(data: dict):
    email = data["email"]
    password = data["password"]
    user = get_user_by_email(conn, email)
    if user is None:
        insert_user(conn, email, password)
        return {
            "signup": True,
            "msg": "User created successfully! Please sign in.",
        }
    else:
        return {"signup": False, "msg": "User already exists with that email!"}
