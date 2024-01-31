import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from db import create_connection
from auth import get_data
import matplotlib.pyplot as plt
import seaborn as sns

# table_name = "plate_detection"
# st.set_page_config(
#     page_title="Car License Plate Recognition Dashboard",
#     layout="wide",
#     initial_sidebar_state="expanded")

conn = create_connection()

alt.themes.enable("dark")

def visualize_car_plate_detection(data):
    st.title("Car License Plate Recognition Dashboard")
    # Display the raw data
    st.subheader("Raw Data")
    st.dataframe(data)

    # Visualize bounding box scores
    st.subheader("Bounding Box Scores Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['bbox_score'], kde=True, ax=ax)
    ax.set_xlabel("Bounding Box Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Visualize text scores
    st.subheader("Text Scores Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['text_score'], kde=True, ax=ax)
    ax.set_xlabel("Text Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# st.write(f"Table Content {table_name}:")
# for row in data:
#     st.write(row)



data = get_data(conn)

    # Call the visualization function with fetched data
visualize_car_plate_detection(data)