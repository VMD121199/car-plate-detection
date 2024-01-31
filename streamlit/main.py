import streamlit as st
from webapp import user_authentication
from streamlit_app import dashboard


def navigation():
    if hasattr(st.session_state, "logged_in") and st.session_state.logged_in:
        dashboard()
    else:
        user_authentication()


# Run the navigation function
if __name__ == "__main__":
    navigation()
