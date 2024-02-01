import json
import requests
import streamlit as st

# from api.auth import create_users_table, insert_user, get_user_by_email
# from db import create_connection


def user_authentication():
    # conn = create_connection()
    # create_users_table(conn)

    st.title("Car plate Detection APP")

    page = st.sidebar.radio("Navigation", ["Sign Up", "Sign In"])

    if page == "Sign Up":
        st.subheader("Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password == confirm_password:
                print(email, password, confirm_password)
                signup_url = "http://localhost:8000/signup/"
                data = {"email": email, "password": password}
                response = requests.post(signup_url, json=data)
                if response.json().get("signup"):
                    st.success(response.json().get("msg"))
                else:
                    st.error(response.json().get("msg"))
            else:
                st.error("Passwords do not match!")

    elif page == "Sign In":
        st.subheader("Sign In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            login_url = "http://localhost:8000/login/"
            data = {"email": email, "password": password}
            response = requests.post(login_url, json=data)
            print(response.json())
            if response.json().get("login"):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.experimental_set_query_params(logged_in=True)
                st.experimental_rerun()
                st.success(response.json().get("msg"))
            else:
                st.error(response.json().get("msg"))


# if __name__ == "__main__":
#     user_authentication()
