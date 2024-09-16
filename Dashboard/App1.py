from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.impute import SimpleImputer

# Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Telecom Data Analysis Dashboard')
st.image(Image.open("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Connect.png"), output_format='auto') 

# Sidebar for file upload
st.sidebar.title("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    df = df.dropna(thresh=len(df.columns) * 0.8)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def plot_data(df):
    st.subheader("Pairplot for Sample Data")
    sns.pairplot(df.sample(min(6000, len(df))), diag_kind='kde', height=4)
    st.pyplot()

    st.subheader("Heatmap of Correlations")
    cols = st.multiselect("Select columns for heatmap:", df.columns.tolist(), default=df.columns.tolist())
    data_corr = df[cols]
    plt.figure(figsize=[10, 8])
    sns.heatmap(data_corr.corr(), annot=True, square=False)
    st.pyplot()

def plot_scaled_data():
    original_data = pd.DataFrame(np.random.exponential(200, size=2000))
    st.write("### Original Data Sample")
    st.write(original_data.sample(5))

    minmax_scaler = MinMaxScaler()
    scaled_data = minmax_scaler.fit_transform(original_data)

    normalizer = Normalizer()
    normalized_data = normalizer.fit_transform(original_data)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    sns.histplot(original_data, ax=ax[0])
    ax[0].set_title("Original Data")
    sns.histplot(scaled_data, ax=ax[1])
    ax[1].set_title("Scaled Data")
    st.pyplot(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    sns.histplot(original_data, ax=ax[0])
    ax[0].set_title("Original Data")
    sns.histplot(normalized_data, ax=ax[1])
    ax[1].set_title("Normalized Data")
    st.pyplot(fig)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("### Original Data Preview")
        st.write(df.head())

        clean_data = preprocess_data(df)
        st.write("### Cleaned Data Preview")
        st.write(clean_data.head())

        clean_data = scale_data(clean_data)
        st.write("### Scaled Data Preview")
        st.write(clean_data.head())

        plot_data(clean_data)
        plot_scaled_data()
else:
    st.write("Please upload a CSV or Excel file to begin.")
