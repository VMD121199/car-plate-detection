import io
import os
import tempfile
import streamlit as st
import requests
import cv2
import numpy as np


def predict_video(video_file):
    fastapi_url = "http://localhost:8000/prediction/"

    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
    response = requests.post(fastapi_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def preprocess_frame(frame, target_size=(300, 300)):
    return cv2.resize(frame, target_size)


# def draw_rectangles_on_frame(frame, bounding_box):
#     x_min, y_min, x_max, y_max = bounding_box
#     return cv2.rectangle(
#         frame.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
#     )


def draw_rectangles_on_frame(frame, bounding_box):
    if frame is not None:
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(
            frame,
            (int(bounding_box[0]), int(bounding_box[1])),
            (int(bounding_box[2]), int(bounding_box[3])),
            color,
            thickness,
        )
    return frame


def display_frames_with_rectangles(video_file, video_result):
    results = video_result.get("results", [])
    video_bytes = video_file.read()

    # Create a temporary file to save the processed video
    fd, temp_file_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as temp_file:
        temp_file.write(video_bytes)

    # Check the number of frames in the video
    cap = cv2.VideoCapture(temp_file_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total Frames:", total_frames)

    if not results:
        st.warning("No results to display.")
        return

    results = results if isinstance(results, list) else [results]

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw rectangles on the frame
        if (
            results[idx].get("license_plate", {}).get("bounding_box")
            and results[idx].get("license_plate", {}).get("bbox_score") >= 0.5
            and results[idx].get("license_plate", {}).get("text_score") >= 0.5
        ):
            frame_with_rectangles = draw_rectangles_on_frame(
                frame,
                results[idx].get("license_plate", {}).get("bounding_box"),
                results[idx].get("license_plate", {}).get("text"),
            )
            if frame_with_rectangles is not None:
                # Display the frame with rectangles using st.image
                st.image(frame_with_rectangles, channels="BGR")
        idx += 1

    cap.release()


def predict_image(image_file):
    fastapi_url = "http://localhost:8000/prediction/"

    files = {"file": ("image.jpg", image_file.getvalue(), "image/jpeg")}
    response = requests.post(fastapi_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def draw_rectangles_on_frame(image, bounding_box, text):
    x_min, y_min, x_max, y_max = np.array(bounding_box, dtype=int)
    detected_image = image.copy()
    cv2.rectangle(
        detected_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3
    )
    cv2.putText(
        detected_image,
        str(text),
        (x_min, y_min - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3,
    )

    return detected_image


def crop_image(image, target_size=(300, 300)):
    resized_image = cv2.resize(image, target_size)
    return resized_image


def dashboard():
    st.title("Prediction Dashboard")

    prediction_type = st.selectbox(
        "Select Prediction Type", ["Video", "Image"]
    )

    if prediction_type == "Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if video_file is not None:
            st.video(video_file)
            if st.button("Get Prediction"):
                video_result = predict_video(video_file)
                if video_result:
                    display_frames_with_rectangles(video_file, video_result)
                else:
                    st.error("Error getting prediction. Please try again.")

    elif prediction_type == "Image":
        image_file = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
        )
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image")

            if st.button("Get Prediction"):
                results = predict_image(image_file)
                if results:
                    st.subheader("Prediction Results:")
                    if "license_plate" in results:
                        st.subheader("Plate Detection:")

                        image_array = np.frombuffer(
                            image_file.getvalue(), dtype=np.uint8
                        )
                        original_image = cv2.imdecode(
                            image_array, cv2.IMREAD_COLOR
                        )
                        original_image_rgb = cv2.cvtColor(
                            original_image, cv2.COLOR_BGR2RGB
                        )

                        image_with_rectangles = draw_rectangles_on_frame(
                            original_image_rgb,
                            results["license_plate"]["bounding_box"],
                            results["license_plate"]["text"],
                        )
                        st.image(
                            image_with_rectangles,
                            caption="Cropped Image with Rectangles",
                        )

                else:
                    st.error("Error getting prediction. Please try again.")


if __name__ == "__main__":
    dashboard()
