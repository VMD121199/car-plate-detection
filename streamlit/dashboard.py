import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from db import create_connection
from auth import get_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time 

conn = create_connection()
st.set_page_config(
    page_title="Car plate detection Dashboard",
    layout="wide",
)

alt.themes.enable("dark")

def visualize_car_plate_detection(data):

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

    time.sleep(1)













    # st.subheader("Past Predictions Data")
    # st.dataframe(data)
    # fig_col1, fig_col2 = st.columns(2)
    # with fig_col1:
    #     st.subheader("Bounding Box Scores Distribution")
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     sns.histplot(data['bbox_score'], kde=True, ax=ax)
    #     ax.set_xlabel("Bounding Box Score")
    #     ax.set_ylabel("Frequency")
    #     st.pyplot(fig)

    #     st.subheader("Text Scores Distribution")
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     sns.histplot(data['text_score'], kde=True, ax=ax)
    #     ax.set_xlabel("Text Score")
    #     ax.set_ylabel("Frequency")
    #     st.pyplot(fig)

    # # Assuming the region information is part of the license plate text
    # # You may need to adjust this logic based on the actual structure of your license plate data
    #     data['region'] = data['license_text'].apply(lambda x: extract_region(x))

    # # Filter data for the specific region
    #     specific_region_data = data[data['region'] == 'uk']

    # # Count the number of cars from the specific region
    #     cars_count = len(specific_region_data)
    #     print(cars_count)

    # with fig_col2:
    #     st.subheader("Bounding Box Density Heatmap")
    #     fig, ax = plt.subplots(figsize=(10, 8)) 
    #     heatmap, xedges, yedges = np.histogram2d(data['x_min'], data['y_min'], bins=50)
    #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #     sns.heatmap(heatmap.T, cmap="YlGnBu", ax=ax)
    #     ax.set_xlabel("x_min")
    #     ax.set_ylabel("y_min")
    #     ax.set_xticks(np.linspace(extent[0], extent[1], num=len(xedges)))
    #     ax.set_yticks(np.linspace(extent[2], extent[3], num=len(yedges)))
    #     ax.set_title("Bounding Box Density Heatmap")
    #     ax.set_title("Bounding Box Density Heatmap")
    #     st.pyplot(fig)

    #     st.subheader("Time Series Plot (Text Score as Actual Detection Time)")
    #     fig, ax = plt.subplots(figsize=(12, 6)) 
    #     ax.plot(data['text_score'], marker='o', linestyle='-')
    #     ax.set_xlabel("Index")
    #     ax.set_ylabel("Text Score (Detection Time)")
    #     st.pyplot(fig)


data = get_data(conn)

visualize_car_plate_detection(data)