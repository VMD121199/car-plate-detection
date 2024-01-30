import io
import streamlit as st
import requests
import cv2
import numpy as np


def predict_video(video_file):
    fastapi_url = "http://localhost:8000/prediction/"

    # Send the file content as bytes
    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
    response = requests.post(fastapi_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def preprocess_frame(frame, target_size=(300, 300)):
    return cv2.resize(frame, target_size)


def draw_rectangles_on_frame(frame, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    return cv2.rectangle(
        frame.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
    )


def display_video_with_rectangles(video_bytes):
    cap = cv2.VideoCapture(io.BytesIO(video_bytes))

    # Get video properties
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a VideoWriter object to save the output video
    out_buffer = io.BytesIO()
    out = cv2.VideoWriter(
        out_buffer, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 300x300
        resized_frame = preprocess_frame(frame)

        # Draw rectangles on the resized frame (Replace this with your object detection logic)
        # For now, we are just using a placeholder bounding box for illustration.
        fake_bounding_box = (50, 50, 200, 200)
        frame_with_rectangles = draw_rectangles_on_frame(
            resized_frame, fake_bounding_box
        )

        # Resize the frame back to its original size
        frame_with_rectangles = cv2.resize(
            frame_with_rectangles, (width, height)
        )

        # Write the frame with rectangles to the output video
        out.write(frame_with_rectangles)

    cap.release()
    out.release()

    # Display the output video in Streamlit
    st.video(out_buffer.getvalue(), format="video/mp4")


def predict_image(image_file):
    fastapi_url = "http://localhost:8000/prediction/"

    # Send the file content as bytes
    files = {"file": ("image.jpg", image_file.getvalue(), "image/jpeg")}
    response = requests.post(fastapi_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def draw_rectangles_on_image(image, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    return cv2.rectangle(
        image.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
    )


def crop_image(image, bounding_box, target_size=(300, 300)):
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
                    display_video_with_rectangles(video_result)
                else:
                    st.error("Error getting prediction. Please try again.")

    elif prediction_type == "Image":
        image_file = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
        )
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image", width=300)

            if st.button("Get Prediction"):
                results = predict_image(image_file)
                if results:
                    st.subheader("Prediction Results:")
                    if "bounding_box" in results:
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

                        cropped_image = crop_image(
                            original_image_rgb, results["bounding_box"]
                        )
                        image_with_rectangles = draw_rectangles_on_image(
                            cropped_image, results["bounding_box"]
                        )
                        st.image(
                            image_with_rectangles,
                            caption="Cropped Image with Rectangles",
                            width=300,
                        )

                else:
                    st.error("Error getting prediction. Please try again.")


if __name__ == "__main__":
    dashboard()
