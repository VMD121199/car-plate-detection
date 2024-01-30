from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from keras.models import load_model
import tempfile

app = FastAPI()
model = load_model("../model/ssd_base.h5")


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


@app.post("/prediction")
async def prediction(file: UploadFile):
    if file.content_type.startswith("video"):
        video_bytes = await file.read()
        # Save the video file to a temporary file
        temp_file_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(video_bytes)

        cap = cv2.VideoCapture(temp_file_path)

        results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = preprocess_frame(frame)
            frame_results = detect_objects_on_frame(processed_frame)
            results.append(frame_results.tolist())

        cap.release()

        return {"results": results}
    elif file.content_type.startswith("image"):
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        processed_image = preprocess_frame(image)
        image_results = detect_objects_on_frame(processed_image)
        print(image_results)
        return {"bounding_box": image_results.tolist()}

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a video or an image.",
        )
