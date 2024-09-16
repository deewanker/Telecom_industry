from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set up page layout
st.set_page_config(layout="wide")

# Set up the page
st.title("Telecom Satisfaction Data Analysis Dashboard")
st.image(Image.open("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Connect.png"))


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Notebook/cleaned_data.csv")

data = load_data()

# Compute Engagement Metrics
engagement_metrics = data.groupby('MSISDN/Number').agg({
    'Dur. (ms)': ['count', 'sum'],
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

engagement_metrics.columns = ['MSISDN/Number', 'Session Frequency', 'Total Duration', 'Total DL', 'Total UL']

# Compute Experience Metrics
experience_metrics = data.groupby('MSISDN/Number').agg({
    'TCP DL Retrans. Vol (Bytes)': 'sum',
    'TCP UL Retrans. Vol (Bytes)': 'sum',
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Handset Type': lambda x: x.mode().iloc[0],
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean'
}).reset_index()

# Merge Engagement and Experience Metrics
satisfaction_analysis = pd.merge(engagement_metrics, experience_metrics, on='MSISDN/Number')

# Create new features
def new_features(ss):
    cols = ['MSISDN/Number', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
            'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    ss['TCP Retransmission'] = ss[cols[5]] + ss[cols[6]]
    ss['RTT'] = ss[cols[1]] + ss[cols[2]]
    ss['Throughput'] = ss[cols[3]] + ss[cols[4]]
    return ss

dt = new_features(data)
dt['TCP Retransmission'] = dt['TCP Retransmission'].fillna(dt['TCP Retransmission'].mean())
dt['RTT'] = dt['RTT'].fillna(dt['RTT'].mean())
dt['Throughput'] = dt['Throughput'].fillna(dt['Throughput'].mean())
dt['Handset Type'] = dt['Handset Type'].fillna(dt['Handset Type'].mode()[0])
aggregate = {'Handset Type': 'first', 'TCP Retransmission': 'sum', 'Throughput': 'sum', 'RTT': 'sum'}
dt = dt.groupby('MSISDN/Number').agg(aggregate).reset_index()
customer_agg = dt.groupby('MSISDN/Number').agg({
    'TCP Retransmission': 'mean',
    'RTT': 'mean',
    'Throughput': 'mean',
    'Handset Type': lambda x: x.mode()[0]
}).reset_index()

# Define distance functions
def compute_engagement_score(user_data, centroid):
    return euclidean_distances([user_data], [centroid])[0][0]

def compute_experience_score(user_data, centroid):
    return euclidean_distances([user_data], [centroid])[0][0]

# K-Means Clustering for Engagement Metrics
kmeans_engagement = KMeans(n_clusters=3, random_state=42).fit(customer_agg[['TCP Retransmission', 'RTT', 'Throughput']])
engagement_centroids = kmeans_engagement.cluster_centers_
least_engaged_cluster = np.argmin(np.sum(engagement_centroids, axis=1))
least_engaged_centroid = engagement_centroids[least_engaged_cluster]
customer_agg['Engagement Score'] = customer_agg.apply(
    lambda row: compute_engagement_score(
        [row['TCP Retransmission'], row['RTT'], row['Throughput']],
        least_engaged_centroid
    ),
    axis=1
)

# K-Means Clustering for Experience Metrics
kmeans_experience = KMeans(n_clusters=3, random_state=42).fit(customer_agg[['TCP Retransmission', 'RTT', 'Throughput']])
experience_centroids = kmeans_experience.cluster_centers_
worst_experience_cluster = np.argmax(np.sum(experience_centroids, axis=1))
worst_experience_centroid = experience_centroids[worst_experience_cluster]
customer_agg['Experience Score'] = customer_agg.apply(
    lambda row: compute_experience_score(
        [row['TCP Retransmission'], row['RTT'], row['Throughput']],
        worst_experience_centroid
    ),
    axis=1
)

# Calculate Satisfaction Score
customer_agg['Satisfaction Score'] = (customer_agg['Engagement Score'] + customer_agg['Experience Score']) / 2
top_10_satisfied = customer_agg.nlargest(10, 'Satisfaction Score')

# Regression Model
features = customer_agg[['TCP Retransmission', 'RTT', 'Throughput']]
target = customer_agg['Satisfaction Score']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# K-Means Clustering on Engagement and Experience Scores
eng_exp_scores = customer_agg[['Engagement Score', 'Experience Score']]
kmeans = KMeans(n_clusters=2, random_state=42)
customer_agg['Engagement-Experience Cluster'] = kmeans.fit_predict(eng_exp_scores)

# Aggregation of Average Scores per Cluster
cluster_agg = customer_agg.groupby('Engagement-Experience Cluster').agg({
    'Satisfaction Score': 'mean',
    'Experience Score': 'mean'
}).reset_index()

# Streamlit Dashboard
st.title("Telecom User Data Analysis Dashboard")

# Sidebar for user options
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select Analysis Type', 
                           ['Overview', 'Top 10 Satisfied Customers', 'Regression Model', 'Clustering Analysis'])

# Overview Tab
if options == 'Overview':
    st.header('Data Overview')
    st.write(data.head())
    st.write(data.describe())

    st.header('Satisfaction Analysis Overview')
    st.write(satisfaction_analysis.head())
    st.write(f"Missing Values in Dataset: {satisfaction_analysis.isnull().sum()}")

# Top 10 Satisfied Customers Tab
elif options == 'Top 10 Satisfied Customers':
    st.header('Top 10 Satisfied Customers')
    st.write(top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']])
    
    st.header('Top 10 Satisfied Customers Plot')
    plt.figure(figsize=(15, 8))
    sns.barplot(
        x='Satisfaction Score',
        y='MSISDN/Number',
        data=top_10_satisfied,
        palette='coolwarm'
    )
    plt.title('Top 10 Satisfied Customers')
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Customer ID')
    st.pyplot(plt.gcf())

# Regression Model Tab
elif options == 'Regression Model':
    st.header('Regression Model Evaluation')
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Model Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")

    st.header('Actual vs Predicted Satisfaction Score')
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Satisfaction Score')
    plt.xlabel('Actual Satisfaction Score')
    plt.ylabel('Predicted Satisfaction Score')
    st.pyplot(plt.gcf())

# Clustering Analysis Tab
elif options == 'Clustering Analysis':
    st.header('Clustering Analysis')

    st.subheader('Engagement and Experience Clustering')
    st.write(f"Cluster Centers for Engagement and Experience Scores:\n{kmeans.cluster_centers_}")

    st.subheader('Clusters Plot')
    sns.scatterplot(
        x='Engagement Score',
        y='Experience Score',
        hue='Engagement-Experience Cluster',
        palette='Set2',
        data=customer_agg,
        s=100,
        alpha=0.7
    )
    plt.title('Clusters of Engagement and Experience Scores')
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.legend(title='Cluster')
    st.pyplot(plt.gcf())

    st.subheader('Average Scores per Cluster')
    st.write(cluster_agg)
    st.bar_chart(cluster_agg.set_index('Engagement-Experience Cluster'))

# Footer
st.sidebar.info("Telecom User Data Analysis Dashboard")
