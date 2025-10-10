import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO

# Streamlit UI
st.title("Saif Check Anomalies")
st.write("Upload an Excel file to detect anomalies")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx","xls"])

if uploaded_file:
    # Process the file
    df = pd.read_excel(uploaded_file)

    # Remove string columns
    df = df.select_dtypes(include=[int, float])

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Fit Isolation Forest model
    clf = IsolationForest(contamination=0.15, random_state=42)
    clf.fit(scaled_data)
    predictions = clf.predict(scaled_data)

    # Identify anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    anomalies = df.iloc[anomaly_indices]

    # Display the number of anomalies
    num_anomalies = len(anomalies)
    st.subheader(f"Number of anomalies detected: {num_anomalies}")

    # Display anomalies
    st.subheader("Anomalies Detected")
    st.write(anomalies)

    # Generate and display graphs
    for col in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df.index, df[col], label="Data")
        ax.scatter(anomaly_indices, df[col].iloc[anomaly_indices], color='red', label="Anomalies")
        ax.set_title(f"Anomalies in {col}")
        ax.legend()
        st.pyplot(fig)