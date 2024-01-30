# main.py
import streamlit as st
from webapp import user_authentication
from streamlit_app import dashboard

def navigation():
    # Check if the user is logged in using session_state
    if not hasattr(st.session_state, 'logged_in') or not st.session_state.logged_in:
        user_authentication()
    else:
        dashboard()

# Run the navigation function
if __name__ == "__main__":
    navigation()
