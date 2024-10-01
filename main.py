import streamlit as st
import pandas as pd
import plotly.express as px
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')
api = KaggleApi()
api.authenticate()

# Download Titanic dataset
@st.cache_data
def load_data():
    api.dataset_download_files('titanic/titanic', path='.', unzip=True)
    return pd.read_csv('train.csv')

# Load the data
df = load_data()

# Streamlit app
st.title("Titanic Dataset Analysis")

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
st.sidebar.info("This Streamlit app analyzes the Titanic dataset from Kaggle. It provides basic information about the dataset, visualizations, and filtering options to explore the data.")
