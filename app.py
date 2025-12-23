import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
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

    # Generate and display interactive charts
    st.subheader("Interactive Charts")
    st.write("Hover over points to see values, zoom in/out, and toggle data series on/off")
    
    for col in df.columns:
        fig = go.Figure()
        
        # Add normal data points
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines+markers',
            name='Normal Data',
            line=dict(color='blue'),
            marker=dict(size=4),
            hovertemplate='<b>Index:</b> %{x}<br><b>Value:</b> %{y}<br><extra></extra>'
        ))
        
        # Add anomaly points
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=df[col].iloc[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='diamond'),
                hovertemplate='<b>Anomaly Index:</b> %{x}<br><b>Value:</b> %{y}<br><extra></extra>'
            ))
        
        # Update layout for better appearance
        fig.update_layout(
            title=f'Anomaly Detection for {col}',
            xaxis_title='Data Point Index',
            yaxis_title=col,
            hovermode='closest',
            showlegend=True,
            height=500
        )
        
        # Display the interactive chart
        st.plotly_chart(fig, use_container_width=True)