import streamlit as st
import hashlib
from auth import create_users_table, insert_user, get_user_by_email
from db import create_connection
# from streamlit_app import dashboard


def get_session_id():
    # Create a unique session ID using the current script code
    script_code = """
    import streamlit as st
    """
    return hashlib.md5(script_code.encode()).hexdigest()

class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def user_authentication():
    # Retrieve or create SessionState
    session_id = get_session_id()
    session_state = st.session_state.setdefault(session_id, SessionState())

    if not hasattr(session_state, "is_authenticated"):
        session_state.is_authenticated = False

    conn = create_connection()
    table_name = "users"
    create_users_table(conn)

    st.title("Car License Plate Recognition App")

    if session_state.is_authenticated:
        # dashboard()
        st.subheader("Sign In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            user = get_user_by_email(conn, table_name, email)
            if user is not None:
                stored_password = user[2]
                if password == stored_password:
                    st.success(f"Logged in as {email}")
                    session_state.is_authenticated = True
                    session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Incorrect password!")
            else:
                st.error("User does not exist!")
    else:
        page = st.sidebar.radio("Navigation", ["Sign Up", "Sign In"])

        if page == "Sign Up":
            st.subheader("Sign Up")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input(
                "Confirm Password", type="password"
            )

            if st.button("Sign Up"):
                if password == confirm_password:
                    existing_user = get_user_by_email(conn, table_name, email)
                    if existing_user:
                        st.error("User already exists with that email!")
                    else:
                        insert_user(conn, table_name, email, password)
                        st.success(
                            "User created successfully! Please sign in."
                        )
                        session_state.is_authenticated = True
                        session_state.user_email = email
                        st.rerun()
                else:
                    st.error("Passwords do not match!")

        elif page == "Sign In":
            st.subheader("Sign In")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Sign In"):
                user = get_user_by_email(conn, table_name, email)
                if user is not None:
                    stored_password = user[2]
                    if password == stored_password:
                        st.success(f"Logged in as {email}")
                        session_state.is_authenticated = True
                        session_state.user_email = email
                        st.rerun()
                    else:
                        st.error("Incorrect password!")
                else:
                    st.error("User does not exist!")
    
    
    # if session_state.is_authenticated == "Sign Up":
    #     st.session_state.current_page = "Sign Up"
    #     if st.button("Sign Up"):
    #         session_state.is_authenticated = "Sign In"
    #         st.rerun()

    
    # elif session_state.show_page == "Sign In":
    #     st.session_state.current_page = "Sign In"
    #     st.write("This is the sign-in page.")
    #     if st.button("Sign In"):
    #         session_state.show_page = "Sign Up"
    #         st.experimental_rerun()


if __name__ == "__main__":
    user_authentication()
