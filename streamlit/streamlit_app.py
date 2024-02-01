import os
import tempfile
import time
import streamlit as st
import requests
import cv2
import numpy as np
import plotly.express as px
from auth import get_data
from db import create_connection


def predict_video(video_file):
    fastapi_url = "http://localhost:8000/prediction/"

    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
    response = requests.post(fastapi_url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        return None


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
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Total Frames:", total_frames)
    width = int(cap.get(3))
    height = int(cap.get(4))
    if not results:
        st.warning("No results to display.")
        return

    results = results if isinstance(results, list) else [results]
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Draw rectangles on the frame
        if len(results[idx]) > 0:
            frame_with_rectangles = frame.copy()
            for rs in results[idx]:
                frame_with_rectangles = draw_rectangles_on_frame(
                    frame_with_rectangles,
                    rs.get("license_plate", {}).get("bounding_box"),
                    rs.get("license_plate", {}).get("text"),
                )
            if frame_with_rectangles is not None:
                # Display the frame with rectangles using st.image
                # st.image(frame_with_rectangles, channels="BGR")
                frames.append(frame_with_rectangles)
        else:
            frames.append(frame)
        idx += 1
    cap.release()
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(fps / 2),
        (width, height),
    )
    for fr in frames:
        out.write(fr)
    out.release()
    st.text(f"Video saved in output.mp4")
    st.video("output.mp4")


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

                    for rs in results:
                        if "license_plate" in rs:
                            original_image_rgb = draw_rectangles_on_frame(
                                original_image_rgb,
                                rs["license_plate"]["bounding_box"],
                                rs["license_plate"]["text"],
                            )
                    st.image(
                        original_image_rgb,
                        caption="Cropped Image with Rectangles",
                    )

                else:
                    st.error("Error getting prediction. Please try again.")

    if st.button("Logout"):
        # Reset session state attributes on logout
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.experimental_rerun()

def visualize_car_plate_detection():
    conn = create_connection()
    data = get_data(conn)

    st.title("Car License Plate Recognition Dashboard")

    st.markdown("### Past Predictions")
    st.dataframe(data)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        fig = px.density_heatmap(data_frame=data, x='x_min', y='y_min', z='bbox_score',
            title='Density Heatmap of Bounding Box Scores',
            labels={'x_min': 'X_min', 'y_min': 'Y_min', 'bbox_score': 'Bounding Box Score'})
        st.write(fig)
    with fig_col2:
        fig = px.scatter(data, x='bbox_score', y='text_score', 
                     title='Scatter Plot of Bounding Box Score vs. Text Score',
                     labels={'bbox_score': 'Bounding Box Score', 'text_score': 'Text Score'})

        st.plotly_chart(fig)



if __name__ == "__main__":
    dashboard()
