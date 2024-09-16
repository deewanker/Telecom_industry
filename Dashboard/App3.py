# Import necessary libraries
from PIL import Image
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Define the function to aggregate data for each application
def total_data(t2):
    cols = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']
    
    t2['Social Media'] = t2[cols[0]] + t2[cols[1]]
    t2['Google'] = t2[cols[2]] + t2[cols[3]]
    t2['Email'] = t2[cols[4]] + t2[cols[5]]
    t2['Youtube'] = t2[cols[6]] + t2[cols[7]]
    t2['Netflix'] = t2[cols[8]] + t2[cols[9]]
    t2['Gaming'] = t2[cols[10]] + t2[cols[11]]
    t2['Other'] = t2[cols[12]] + t2[cols[13]]
    t2['Total'] = t2[cols[14]] + t2[cols[15]]
    
    return t2

# Set up page layout
st.set_page_config(layout="wide")

# Set up the page
st.title("Telecom User Engagement Dashboard")
st.image(Image.open("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Connect.png"))

# Load data
t2 = pd.read_csv("C:/Users/vipin/OneDrive/Documents/Desktop/Project5/Notebook/cleaned_data.csv")

# Data Cleaning
# Convert non-numeric columns to numeric, forcing errors to NaN
for col in t2.columns:
    t2[col] = pd.to_numeric(t2[col], errors='coerce')

# Title
st.title('User Engagement Dashboard')

# Sidebar for user options
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select Analysis Type', 
                           ['Overview', 'User Engagement Clustering', 'Application Usage', 'Top Users'])

# Overview Tab
if options == 'Overview':
    st.header('Data Overview')
    st.write(t2.head())
    
    st.header('Data Statistics')
    st.write(t2.describe())

    # Heatmap
    st.header('Correlation Heatmap')
    plt.figure(figsize=(12, 7))
    sns.heatmap(t2.corr(), annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=False)
    st.pyplot(plt.gcf())

# User Engagement Clustering Tab
elif options == 'User Engagement Clustering':
    st.header('User Engagement Clustering')
    
    # Aggregate data
    t2['TotalTraffic'] = t2['Total DL (Bytes)'] + t2['Total UL (Bytes)']
    agg_data = t2.groupby('MSISDN/Number').agg({
        'Dur. (ms)': ['count', 'sum', 'mean'],
        'TotalTraffic': 'sum'
    }).reset_index()
    
    agg_data.columns = ['MSISDN', 'SessionFrequency', 'TotalSessionDuration', 'AverageSessionDuration', 'TotalTraffic']
    
    # Normalize data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(agg_data[['SessionFrequency', 'TotalSessionDuration', 'TotalTraffic']])
    
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    agg_data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    st.write("K-Means Clustering Result")
    st.write(agg_data.head())
    
    # Plot pairplot of clusters
    st.header('Pair Plot of Clusters')
    sns.pairplot(agg_data, hue='Cluster', vars=['SessionFrequency', 'TotalSessionDuration', 'TotalTraffic'], palette='viridis')
    st.pyplot(plt.gcf())
    
    # Show cluster statistics
    st.header('Cluster Statistics')
    cluster_stats = agg_data.groupby('Cluster').agg({
        'SessionFrequency': ['min', 'max', 'mean', 'sum'],
        'TotalSessionDuration': ['min', 'max', 'mean', 'sum'],
        'TotalTraffic': ['min', 'max', 'mean', 'sum']
    })
    st.write(cluster_stats)

# Application Usage Tab
elif options == 'Application Usage':
    st.header('Application Usage Analysis')

    # Process data for application usage
    data = total_data(t2)

    # Aggregate traffic by app
    app_traffic = data[['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']].sum().reset_index()
    app_traffic.columns = ['Application', 'TotalTraffic']

    # Bar Plot of Total Traffic per App
    st.header('Total Traffic per Application')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=app_traffic['Application'], y=app_traffic['TotalTraffic'], palette='viridis')
    plt.title('Total Traffic by Application')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')
    st.pyplot(plt.gcf())

    # Pie chart of top 3 applications
    st.header('Top 3 Applications by Traffic')
    top_3_apps = app_traffic.nlargest(3, 'TotalTraffic')
    labels = top_3_apps['Application']
    sizes = top_3_apps['TotalTraffic']
    colors = sns.color_palette('viridis', len(top_3_apps))

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title('Top 3 Applications by Traffic')
    st.pyplot(plt.gcf())

# Top Users Tab
elif options == 'Top Users':
    st.header('Top Users by Application')

    # Ensure the data includes the application columns
    data = total_data(t2) 
    
    # Check if the required columns are present
    required_columns = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"Missing columns in DataFrame: {missing_columns}")
    else:
        # Melt the DataFrame if all required columns are present
        top_users_per_app = pd.melt(data, id_vars=['MSISDN/Number'], value_vars=required_columns,
                                    var_name='Application', value_name='app_traffic')
        
        # Aggregate and get top 10 users per application
        top_10_users_per_app = top_users_per_app.groupby(['Application', 'MSISDN/Number']).agg({'app_traffic': 'sum'}).reset_index()
        top_10_users_per_app = top_10_users_per_app.groupby('Application').apply(lambda x: x.nlargest(10, 'app_traffic')).reset_index(drop=True)

        st.write("Top 10 Users per Application")
        st.write(top_10_users_per_app)

# Footer
st.sidebar.info("User Engagement Dashboard")
