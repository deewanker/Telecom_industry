# Import necessary libraries
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up the page
st.title("Telecom User Data Analysis Dashboard")
st.image(Image.open("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Connect.png")) 

# Load dataset
@st.cache_data
def load_data():
    tel = pd.read_csv("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Notebook/cleaned_data.csv")
    return tel

tel = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Analysis Section", 
                           ["Overview", "Top Handsets and Manufacturers", "User Behavior Analysis", "Data Correlations", "PCA Analysis"])

# Overview section
if options == "Overview":
    st.header("Data Overview")
    st.write("### First few rows of the dataset")
    st.dataframe(tel.head())

    st.write("### Dataset Info")
    st.text(tel.info())

    st.write("### Dataset Description")
    st.write(tel.describe())

# Top Handsets and Manufacturers Section
elif options == "Top Handsets and Manufacturers":
    st.header("Top Handsets and Manufacturers")

    # Top 10 Handsets
    st.subheader("Top 10 Handsets")
    top_10_handsets = tel['Handset Type'].value_counts().head(10)
    st.bar_chart(top_10_handsets)

    # Top 3 Manufacturers
    st.subheader("Top 3 Handset Manufacturers")
    top_3_manufacturers = tel['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_3_manufacturers)

    # Top 5 Handsets per Manufacturer
    st.subheader("Top 5 Handsets per Manufacturer")
    for manufacturer in top_3_manufacturers.index:
        top_5_handsets = tel[tel['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.write(f"Top 5 Handsets by {manufacturer}")
        st.bar_chart(top_5_handsets)

# User Behavior Analysis Section
elif options == "User Behavior Analysis":
    st.header("User Behavior Analysis")

    # Aggregate user behavior data
    user_behavior = tel.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total UL (Bytes)': 'sum',
        'Total DL (Bytes)': 'sum'
    }).rename(columns={
        'Bearer Id': 'xDR_sessions',
        'Dur. (ms)': 'session_duration',
        'Total UL (Bytes)': 'total_upload',
        'Total DL (Bytes)': 'total_download'
    })

    user_behavior['total_data_volume'] = user_behavior['total_upload'] + user_behavior['total_download']

    st.write("### Aggregated User Behavior Data")
    st.dataframe(user_behavior.head())

    # Descriptive stats
    descriptive_stats = user_behavior.describe()
    st.write("### Descriptive Statistics")
    st.write(descriptive_stats)

# Data Correlations Section
elif options == "Data Correlations":
    st.header("Data Correlations")
    
    # Filter only numeric columns
    numeric_df = tel.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    st.write("### Correlation Matrix")
    st.dataframe(corr_matrix)
    
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# PCA Analysis Section
elif options == "PCA Analysis":
    st.header("PCA Analysis")
    
    # Select numeric columns
    numeric_df = tel.select_dtypes(include='float64')

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply PCA
    pca = PCA(n_components=10)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    # Plot cumulative explained variance
    st.write("### Cumulative Explained Variance by PCA Components")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
    ax.axhline(y=0.99, color='r', linestyle='-')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance vs Number of Components')
    st.pyplot(fig)
