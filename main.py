import streamlit as st
import pandas as pd
import plotly.express as px
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.environ.get('KAGGLE_KEY', '')

@st.cache_data
def load_data():
    st.info("Attempting to load the Titanic dataset...")
    
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
    
    if os.path.exists('train.csv'):
        st.info("Found existing train.csv file. Loading data...")
        return pd.read_csv('train.csv')
    
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
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 21.075, 11.1333, 21.0, 30.0708]
    }
    return pd.DataFrame(sample_data)

df = load_data()

st.title("Titanic Dataset Analysis")

if df is not None:
    st.header("Dataset Information")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:")
    st.write(df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)

    st.header("Passenger Class Distribution")
    fig = px.pie(df, names='Pclass', title='Distribution of Passenger Classes')
    st.plotly_chart(fig)

    st.header("Dataset Preview")
    num_rows = st.slider("Select number of rows to display", 5, 50, 10)
    st.write(df.head(num_rows))

    st.header("Data Filtering")
    selected_columns = st.multiselect("Select columns to display", df.columns.tolist(), default=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare'])

    pclass_filter = st.multiselect("Filter by Passenger Class", df['Pclass'].unique())
    if pclass_filter:
        df_filtered = df[df['Pclass'].isin(pclass_filter)]
    else:
        df_filtered = df

    sex_filter = st.multiselect("Filter by Sex", df['Sex'].unique())
    if sex_filter:
        df_filtered = df_filtered[df_filtered['Sex'].isin(sex_filter)]

    age_range = st.slider("Filter by Age Range", float(df['Age'].min()), float(df['Age'].max()), (float(df['Age'].min()), float(df['Age'].max())))
    df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]

    st.write(df_filtered[selected_columns])

    st.header("Filtered Data Statistics")
    st.write(df_filtered[selected_columns].describe())

    st.header("Enhanced Correlation Analysis")
    numeric_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns
    selected_features = st.multiselect("Select features for correlation analysis", numeric_columns, default=numeric_columns[:5])
    if len(selected_features) > 1:
        correlation_matrix = df_filtered[selected_features].corr()
        fig_corr = px.imshow(correlation_matrix, title="Correlation Heatmap", labels=dict(color="Correlation"))
        st.plotly_chart(fig_corr)
    else:
        st.warning("Please select at least two features for correlation analysis.")

    st.header("Survival Rate by Passenger Class")
    survival_rate = df_filtered.groupby('Pclass')['Survived'].mean()
    fig_survival = px.bar(survival_rate, x=survival_rate.index, y='Survived', title="Survival Rate by Passenger Class")
    st.plotly_chart(fig_survival)

    st.header("Age Distribution")
    fig_age = px.histogram(df_filtered, x='Age', nbins=20, title="Age Distribution")
    st.plotly_chart(fig_age)

    st.header("Feature Importance and Survival Probability")
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    cat_cols = ['Sex', 'Embarked']
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    X = df[feature_cols].copy()
    y = df['Survived']

    le = LabelEncoder()
    X[cat_cols] = X[cat_cols].apply(lambda col: le.fit_transform(col.astype(str)))

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    X_processed = pd.concat([X[cat_cols], X[num_cols]], axis=1)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_processed, y)

    feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    fig_importance = px.bar(feature_importance, x='feature', y='importance', title="Feature Importance")
    st.plotly_chart(fig_importance)

    st.header("Survival Probability Calculator")
    st.write("Enter passenger details to calculate survival probability:")

    input_pclass = st.selectbox("Passenger Class", [1, 2, 3])
    input_sex = st.selectbox("Sex", ["male", "female"])
    input_age = st.slider("Age", 0, 100, 30)
    input_sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    input_parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
    input_fare = st.slider("Fare", 0.0, 512.0, 32.0)
    input_embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Calculate Survival Probability"):
        input_data = pd.DataFrame({
            'Pclass': [input_pclass],
            'Sex': [input_sex],
            'Age': [input_age],
            'SibSp': [input_sibsp],
            'Parch': [input_parch],
            'Fare': [input_fare],
            'Embarked': [input_embarked]
        })
        
        input_data[cat_cols] = input_data[cat_cols].apply(lambda col: le.transform(col.astype(str)))
        input_data[cat_cols] = cat_imputer.transform(input_data[cat_cols])
        input_data[num_cols] = num_imputer.transform(input_data[num_cols])
        
        input_processed = pd.concat([input_data[cat_cols], input_data[num_cols]], axis=1)
        
        survival_prob = rf_model.predict_proba(input_processed)[0][1]
        st.write(f"Survival Probability: {survival_prob:.2%}")

    st.sidebar.header("About")
    st.sidebar.info("This Streamlit app analyzes the Titanic dataset. It provides basic information about the dataset, visualizations, filtering options, correlation analysis, feature importance, and a survival probability calculator.")
else:
    st.error("Failed to load the Titanic dataset. Please check your internet connection and try again.")