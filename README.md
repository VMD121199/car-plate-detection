# Car Plate Detection

## Introduction
Welcome to our GitHub repository, where we've developed an advanced vehicle tracking and license plate recognition system designed to address the challenges of efficient traffic control, surveillance, and recovery of stolen vehicles. This project leverages cutting-edge deep learning strategies to overcome the complexities of license plate recognition, including variations in background, font color, style, size, and non-standard characters often encountered in developing countries.</br>
Our system is trained and validated on a diverse dataset captured under various lighting conditions, distances, angles, and rotations, ensuring high recognition rates even in the most challenging scenarios. It aims to serve as a valuable tool for law enforcement agencies and private organizations, enhancing homeland security measures.</br>
The repository contains all the necessary code, models, and documentation to deploy and test the system. We invite collaborators to contribute towards further innovations, including the integration of hybrid classifier methods and enhancements for robust performance across different weather conditions.
## Technologies Used
 
This project leverages several cutting-edge technologies:
 
- **Python**: The core programming language used for development. [Python](https://www.python.org/)
- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+ that includes automatic Swagger UI generation. [FastAPI](https://fastapi.tiangolo.com/)
- **Streamlit**: Streamlit turns data scripts into shareable web apps in minutes. It's an open-source app framework specifically for Machine Learning and Data Science projects. [Streamlit](https://streamlit.io/)
- **EasyOCR**: A Python library that simplifies OCR (Optical Character Recognition) and supports multiple languages. [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **YOLO (You Only Look Once)**: For real-time object detection, YOLO algorithms are used for identifying and localizing objects, including license plates in images or video feeds. [YOLO](https://pjreddie.com/darknet/yolo/)
- **Docker** (optional): For containerizing the application, making it easy to deploy and run on any platform. [Docker](https://www.docker.com/)
 
Each technology contributes to the project's goal of efficient, real-time license plate detection and recognition, ensuring high accuracy and performance.
## Installation Guide
 
Follow these steps to install and run the app:
 
1. **Clone the Repository**: `git clone https://github.com/VMD121199/car-plate-detection.git`
2. **Navigate to Project Directory**: `cd car-plate-detection`
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Start FastAPI Server**: `cd api` then `uvicorn main:app --reload`
5. **Launch Streamlit App**: `cd streamlit` then `streamlit run main.py`
 
Ensure Python>=3.9.0 is installed and consider using a virtual environment for dependency management.
