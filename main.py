import streamlit as st
import pandas as pd
import plotly.express as px
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import requests

# Set up Kaggle API using environment variables
os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.environ.get('KAGGLE_KEY', '')

# Function to download Titanic dataset
@st.cache_data
def load_data():
    st.info("Attempting to load the Titanic dataset...")
    
    # Check if we have Kaggle credentials
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        try:
            with st.spinner("Downloading dataset from Kaggle..."):
                api = KaggleApi()
                api.authenticate()
                st.info(f"Authenticated with Kaggle as user: {os.environ.get('KAGGLE_USERNAME')}")
                api.dataset_download_files('titanic/titanic', path='.', unzip=True)
            st.success("Successfully downloaded dataset from Kaggle.")
            return pd.read_csv('train.csv')
        except Exception as e:
            st.warning(f"Error using Kaggle API: {str(e)}")
            st.info("Kaggle authentication failed. Checking for alternative data sources...")
    else:
        st.warning("Kaggle credentials not found in environment variables. Skipping Kaggle download.")
    
    # Fallback: Check if train.csv already exists
    if os.path.exists('train.csv'):
        st.info("Found existing train.csv file. Loading data...")
        return pd.read_csv('train.csv')
    
    # Fallback: Download CSV directly
    st.info("Attempting to download dataset directly...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        with st.spinner("Downloading dataset from GitHub..."):
            response = requests.get(url)
            response.raise_for_status()
            with open('titanic.csv', 'wb') as f:
                f.write(response.content)
        st.success("Successfully downloaded dataset from GitHub.")
        return pd.read_csv('titanic.csv')
    except Exception as e:
        st.error(f"Failed to download dataset from GitHub: {str(e)}")
    
    # Final fallback: Use a small sample dataset
    st.warning("All download attempts failed. Using a small sample dataset for demonstration.")
    sample_data = {
        'PassengerId': range(1, 11),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 2, 3, 3, 2, 2, 1],
        'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'Charlie Davis', 'Eva Wilson', 'Frank Miller', 'Grace Taylor', 'Henry Anderson', 'Ivy Clark'],
        'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female'],
        'Age': [22, 38, 26, 35, 28, 27, 32, 29, 31, 45],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 21.075, 11.1333, 21.0, 30.0708]
    }
    return pd.DataFrame(sample_data)

# Load the data
df = load_data()

# Streamlit app
st.title("Titanic Dataset Analysis")

if df is not None:
    # Display basic information about the dataset
    st.header("Dataset Information")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:")
    st.write(df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)

    # Show a simple visualization of the data
    st.header("Passenger Class Distribution")
    fig = px.pie(df, names='Pclass', title='Distribution of Passenger Classes')
    st.plotly_chart(fig)

    # Allow users to view the first few rows of the dataset
    st.header("Dataset Preview")
    num_rows = st.slider("Select number of rows to display", 5, 50, 10)
    st.write(df.head(num_rows))

    # Implement basic data filtering options
    st.header("Data Filtering")
    selected_columns = st.multiselect("Select columns to display", df.columns.tolist(), default=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare'])

    # Filter by passenger class
    pclass_filter = st.multiselect("Filter by Passenger Class", df['Pclass'].unique())
    if pclass_filter:
        df_filtered = df[df['Pclass'].isin(pclass_filter)]
    else:
        df_filtered = df

    # Filter by sex
    sex_filter = st.multiselect("Filter by Sex", df['Sex'].unique())
    if sex_filter:
        df_filtered = df_filtered[df_filtered['Sex'].isin(sex_filter)]

    # Filter by age range
    age_range = st.slider("Filter by Age Range", float(df['Age'].min()), float(df['Age'].max()), (float(df['Age'].min()), float(df['Age'].max())))
    df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]

    # Display filtered data
    st.write(df_filtered[selected_columns])

    # Show statistics of filtered data
    st.header("Filtered Data Statistics")
    st.write(df_filtered[selected_columns].describe())

    # Correlation heatmap
    st.header("Correlation Heatmap")
    numeric_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df_filtered[numeric_columns].corr()
    fig_corr = px.imshow(correlation_matrix, title="Correlation Heatmap")
    st.plotly_chart(fig_corr)

    # Survival rate by passenger class
    st.header("Survival Rate by Passenger Class")
    survival_rate = df_filtered.groupby('Pclass')['Survived'].mean()
    fig_survival = px.bar(survival_rate, x=survival_rate.index, y='Survived', title="Survival Rate by Passenger Class")
    st.plotly_chart(fig_survival)

    # Age distribution
    st.header("Age Distribution")
    fig_age = px.histogram(df_filtered, x='Age', nbins=20, title="Age Distribution")
    st.plotly_chart(fig_age)

    st.sidebar.header("About")
    st.sidebar.info("This Streamlit app analyzes the Titanic dataset. It provides basic information about the dataset, visualizations, and filtering options to explore the data.")
else:
    st.error("Failed to load the Titanic dataset. Please check your internet connection and try again.")
