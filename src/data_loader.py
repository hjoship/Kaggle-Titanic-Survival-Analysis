import streamlit as st
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    st.info("Attempting to load the Titanic dataset...")
    
    if os.path.exists('titanic.csv'):
        st.info("Found existing titanic.csv file. Loading data...")
        return pd.read_csv('titanic.csv')
    
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
    else:
        st.warning("Kaggle credentials not found in environment variables.")
    
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
    
    st.warning("All download attempts failed. Using a small sample dataset for demonstration.")
    sample_data = {
        'PassengerId': range(1, 11),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 2, 3, 3, 2, 2, 1],
        'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'Charlie Davis', 'Eva Wilson', 'Frank Miller', 'Grace Taylor', 'Henry Anderson', 'Ivy Clark'],
        'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female'],
        'Age': [22, 38, 26, 35, 28, 27, 32, 29, 31, 45],
        'SibSp': [1, 1, 0, 0, 0, 0, 1, 0, 2, 1],
        'Parch': [0, 0, 0, 0, 0, 2, 0, 0, 0, 1],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 21.075, 11.1333, 21.0, 30.0708],
        'Embarked': ['S', 'C', 'S', 'S', 'Q', 'S', 'S', 'S', 'Q', 'C']
    }
    return pd.DataFrame(sample_data)

def preprocess_data(df):
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cat_cols = ['Sex', 'Embarked']
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    X = df[feature_cols].copy()
    y = df['Survived']

    le = LabelEncoder()
    le_sex = LabelEncoder()
    le_sex.fit(df['Sex'])

    X['Embarked'] = le.fit_transform(X['Embarked'].astype(str))
    X['Sex'] = le_sex.transform(X['Sex'])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    X_processed = pd.concat([X[cat_cols], X[num_cols]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le, le_sex, cat_imputer, num_imputer