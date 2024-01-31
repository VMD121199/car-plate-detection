import streamlit as st
import tempfile
import os

# Upload video file or use a default video file
video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
if video_file is not None:
    st.video(video_file)
