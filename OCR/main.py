from ultralytics import YOLO
import cv2
import numpy as np
from datasets import load_dataset

from util import read_license_plate, write_csv

results = {}

# Loading models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("models/best.pt")

# Loading dataset
from datasets import load_dataset

ds = load_dataset("keremberke/license-plate-object-detection", name="full")
images = ds["train"]

for idx, image in enumerate(images):
    results[idx] = {}

    frame = cv2.cvtColor(np.array(image["image"]), cv2.COLOR_RGB2BGR)

    # Detecting license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Cropping license plate
        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

        # Processing license plate
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop, cv2.COLOR_BGR2GRAY
        )
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY
        )

        # Reading license plate number using a function from util
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )

        if license_plate_text is not None:
            results[idx][class_id] = {
                "license_plate": {
                    "bbox": [x1, y1, x2, y2],
                    "text": license_plate_text,
                    "bbox_score": score,
                    "text_score": license_plate_text_score,
                }
            }

write_csv(results, "output_data/test2.csv")
