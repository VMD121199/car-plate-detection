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
    x_min, y_min, x_max, y_max = bounding_box
    resized_image = cv2.resize(image, target_size)
    cropped_image = resized_image[y_min:y_max, x_min:x_max]
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
                results = predict_video(video_file)
                if results:
                    st.subheader("Prediction Results:")
                    if "video_result" in results:
                        st.subheader("Video Result:")
                        st.video(results["video_result"])
                    if "text_result" in results:
                        st.subheader("Text Result:")
                        st.write(results["text_result"])
                else:
                    st.error("Error getting prediction. Please try again.")
    elif prediction_type == "Image":
        image_file = st.file_uploader(
            "Upload Image", type=["jpg", "png", "jpeg"]
        )
        if image_file is not None:
            st.image(
                image_file, caption="Uploaded Image", use_column_width=True
            )

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
                            use_column_width=True,
                            width=300,
                        )

                else:
                    st.error("Error getting prediction. Please try again.")


if __name__ == "__main__":
    dashboard()
