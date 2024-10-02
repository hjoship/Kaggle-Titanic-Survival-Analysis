import streamlit as st
from src.app import *

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Titanic Dataset Analysis")
    st.write("Debug: Starting the app")

    # The main app logic is now in src/app.py
    # This file serves as an entry point for the Streamlit app
