import streamlit as st
import requests

st.title("Video Prediction Dashboard")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if video_file is not None:
    st.video(video_file)
    if st.button("Get Prediction"):
        files = {"video": video_file.getvalue()}
        fastapi_url = "http://your-fastapi-url/predict"
        response = requests.post(fastapi_url, files=files)
        if response.status_code == 200:
            results = response.json()
            st.subheader("Prediction Results:")
            if "video_result" in results:
                st.subheader("Video Result:")
                st.video(results["video_result"])
            if "text_result" in results:
                st.subheader("Text Result:")
                st.write(results["text_result"])
        else:
            st.error("Error getting prediction. Please try again.")
