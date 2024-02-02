import streamlit as st
from webapp import user_authentication
from streamlit_app import dashboard, visualize_car_plate_detection


def navigation():
    if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
        page = st.sidebar.radio("Pages", ["Prediction", "Dashboard"])
        if page == "Prediction":
            dashboard()
        if page == "Dashboard":
            visualize_car_plate_detection()
                # dashboard()
    else:
        user_authentication()


# Run the navigation function
if __name__ == "__main__":
    navigation()
