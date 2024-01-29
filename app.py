# app.py
import streamlit as st
from auth import create_users_table, insert_user, get_user_by_email
from db import create_connection

def main():
    conn = create_connection()
    create_users_table(conn)

    st.title("User Authentication App")

    page = st.sidebar.radio("Navigation", ["Sign Up", "Sign In"])

    if page == "Sign Up":
        st.subheader("Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password == confirm_password:
                existing_user = get_user_by_email(conn, email)
                if existing_user:
                    st.error("User already exists with that email!")
                else:
                    insert_user(conn, email, password)
                    st.success("User created successfully! Please sign in.")
            else:
                st.error("Passwords do not match!")

    elif page == "Sign In":
        st.subheader("Sign In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            user = get_user_by_email(conn, email)
            if user:
                if password == user[2]:
                    st.success(f"Logged in as {email}")
                else:
                    st.error("Incorrect password!")
            else:
                st.error("User does not exist!")

if __name__ == "__main__":
    main()
