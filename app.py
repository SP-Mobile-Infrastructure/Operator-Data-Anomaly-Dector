import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta

# Optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def detect_date_columns(df):
    """Detect columns that contain date/datetime information"""
    date_columns = []
    
    for col in df.columns:
        # Check if column dtype is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
            
        # Try to parse as datetime for object/string columns
        if df[col].dtype == 'object':
            try:
                # Sample a few non-null values to test
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Try to parse sample values
                    parsed_count = 0
                    for val in sample_values:
                        try:
                            pd.to_datetime(val)
                            parsed_count += 1
                        except:
                            continue
                    
                    # If most samples can be parsed as dates, consider it a date column
                    if parsed_count / len(sample_values) >= 0.7:
                        date_columns.append(col)
            except:
                continue
    
    return date_columns

def calculate_feature_contributions(df_scaled, anomaly_indices, feature_names):
    """Calculate how much each feature contributes to anomaly detection"""
    
    # Calculate mean and std for each feature
    feature_means = np.mean(df_scaled, axis=0)
    feature_stds = np.std(df_scaled, axis=0)
    
    contributions = {}
    
    for idx in anomaly_indices:
        row_contributions = {}
        row_data = df_scaled[idx]
        
        for i, feature in enumerate(feature_names):
            # Calculate z-score (how many standard deviations away from mean)
            z_score = abs((row_data[i] - feature_means[i]) / (feature_stds[i] + 1e-8))
            row_contributions[feature] = z_score
        
        contributions[idx] = row_contributions
    
    return contributions

def highlight_anomalous_values(df, contributions, anomaly_indices):
    """Apply styling to highlight anomalous values in the dataframe"""
    
    def apply_styling(row):
        styles = [''] * len(row)
        
        # Get the index of this row in the original dataframe
        row_idx = row.name
        
        if row_idx in contributions:
            row_contributions = contributions[row_idx]
            
            # Sort contributions by value to identify top contributors
            sorted_contribs = sorted(row_contributions.items(), key=lambda x: x[1], reverse=True)
            
            for i, (col_name, contrib_value) in enumerate(sorted_contribs):
                if col_name in row.index:
                    col_idx = row.index.get_loc(col_name)
                    
                    # Apply different colors based on contribution level with better contrast
                    if i == 0 and contrib_value > 2.0:  # Primary contributor
                        styles[col_idx] = 'background-color: #ff4444; color: white; font-weight: bold; border: 2px solid #cc0000'
                    elif i <= 2 and contrib_value > 1.5:  # Secondary contributors
                        styles[col_idx] = 'background-color: #ffa500; color: black; font-weight: bold; border: 1px solid #ff8c00'
                    elif contrib_value > 1.0:  # Minor contributors
                        styles[col_idx] = 'background-color: #add8e6; color: black; border: 1px solid #87ceeb'
        
        return styles
    
    return df.style.apply(apply_styling, axis=1)

def count_daily_transactions(df, date_col):
    """Count transactions per day from the date column"""
    
    # Convert to datetime if not already
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Extract date (without time) for daily counting
    df_copy['date_only'] = df_copy[date_col].dt.date
    
    # Count transactions per day
    daily_counts = df_copy['date_only'].value_counts().sort_index()
    
    # Convert back to datetime for plotting
    daily_counts.index = pd.to_datetime(daily_counts.index)
    
    return daily_counts

def create_transaction_summary_charts(daily_counts):
    """Create visualizations for daily transaction counts"""
    
    if len(daily_counts) == 0:
        st.warning("No valid dates found for transaction counting.")
        return
    
    if HAS_PLOTLY:
        # Create interactive Plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines+markers',
            name='Daily Transactions',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add average line
        avg_transactions = daily_counts.mean()
        fig.add_hline(y=avg_transactions, line_dash="dash", line_color="red", 
                      annotation_text=f"Average: {avg_transactions:.1f}")
        
        fig.update_layout(
            title="Daily Transaction Volume",
            xaxis_title="Date",
            yaxis_title="Number of Transactions",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width='stretch')
    
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(daily_counts.index, daily_counts.values, 'b-', marker='o', 
                linewidth=2, markersize=4, label='Daily Transactions')
        
        # Add average line
        avg_transactions = daily_counts.mean()
        ax.axhline(y=avg_transactions, color='red', linestyle='--', 
                   label=f'Average: {avg_transactions:.1f}')
        
        ax.set_title("Daily Transaction Volume", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Transactions", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{daily_counts.sum():,}")
    with col2:
        st.metric("Average Daily", f"{daily_counts.mean():.1f}")
    with col3:
        st.metric("Peak Day", f"{daily_counts.max():,}")
    with col4:
        st.metric("Minimum Day", f"{daily_counts.min():,}")
    
    # Show top and bottom days
    st.subheader("Transaction Volume Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Highest Volume Days:**")
        top_days = daily_counts.nlargest(5)
        for date, count in top_days.items():
            st.write(f"• {date.strftime('%Y-%m-%d')}: {count:,} transactions")
    
    with col2:
        st.write("**Lowest Volume Days:**")
        bottom_days = daily_counts.nsmallest(5)
        for date, count in bottom_days.items():
            st.write(f"• {date.strftime('%Y-%m-%d')}: {count:,} transactions")

# Streamlit UI
st.title("Saif Check Anomalies")
st.write("Upload an Excel file to detect anomalies and analyze transaction patterns")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Anomaly detection settings
    st.subheader("Anomaly Detection")
    contamination_rate = st.slider(
        "Contamination Rate", 
        min_value=0.05, 
        max_value=0.30, 
        value=0.15, 
        step=0.05,
        help="Expected proportion of anomalies in the dataset"
    )
    
    random_seed = st.number_input(
        "Random Seed", 
        value=42, 
        help="For reproducible results"
    )
    
    # Display options
    st.subheader("Display Options")
    show_transaction_analysis = st.checkbox(
        "Show Transaction Analysis", 
        value=True,
        help="Analyze daily transaction patterns (requires date columns)"
    )
    
    max_anomaly_details = st.slider(
        "Max Anomaly Details to Show", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Number of detailed anomaly breakdowns to display"
    )
    
    show_correlation_heatmap = st.checkbox(
        "Show Correlation Heatmap", 
        value=True,
        help="Display feature correlations for anomalous data"
    )

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx","xls"])

if uploaded_file:
    # Process the file
    original_df = pd.read_excel(uploaded_file)
    st.success(f"File uploaded successfully! Found {len(original_df)} rows and {len(original_df.columns)} columns.")
    
    # Detect date columns before processing
    date_columns = detect_date_columns(original_df)
    
    # Transaction Analysis Section
    if date_columns and show_transaction_analysis:
        st.header("📊 Transaction Volume Analysis")
        
        # Let user select which date column to use
        if len(date_columns) == 1:
            selected_date_col = date_columns[0]
            st.info(f"Using detected date column: **{selected_date_col}**")
        else:
            selected_date_col = st.selectbox(
                "Select date column for transaction analysis:",
                date_columns
            )
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            aggregation_level = st.selectbox(
                "Aggregation Level:",
                ["Daily", "Weekly", "Monthly"],
                index=0
            )
        
        with col2:
            show_weekend_analysis = st.checkbox(
                "Highlight Weekends",
                value=True
            )
        
        # Analyze transaction patterns
        if selected_date_col:
            try:
                daily_counts = count_daily_transactions(original_df, selected_date_col)
                create_transaction_summary_charts(daily_counts)
                
                # Add separator
                st.markdown("---")
                
            except Exception as e:
                st.warning(f"Could not analyze transaction patterns: {str(e)}")
    
    elif date_columns and not show_transaction_analysis:
        st.info("💡 Transaction analysis disabled in settings. Enable it in the sidebar to see transaction volume patterns.")
    else:
        st.info("💡 No date columns detected. Upload a file with date/time columns to see transaction volume analysis.")
    
    # Continue with anomaly detection on numerical data
    st.header("🔍 Anomaly Detection Analysis")
    
    # Remove string columns for anomaly detection
    df = original_df.select_dtypes(include=[int, float])
    
    if len(df.columns) == 0:
        st.error("No numerical columns found for anomaly detection. Please ensure your file contains numerical data.")
        st.stop()
    
    st.info(f"Analyzing {len(df.columns)} numerical columns: {', '.join(df.columns)}")

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Fit Isolation Forest model
    clf = IsolationForest(contamination=contamination_rate, random_state=int(random_seed))
    clf.fit(scaled_data)
    predictions = clf.predict(scaled_data)

    # Identify anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    anomalies = df.iloc[anomaly_indices]

    # Calculate feature contributions for highlighting
    contributions = calculate_feature_contributions(scaled_data, anomaly_indices, df.columns)

    # Display the number of anomalies
    num_anomalies = len(anomalies)
    
    # Show configuration summary
    with st.expander("🔧 Analysis Configuration"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Contamination Rate", f"{contamination_rate:.1%}")
        with col2:
            st.metric("Random Seed", int(random_seed))
        with col3:
            st.metric("Features Analyzed", len(df.columns))
    
    st.subheader(f"Number of anomalies detected: {num_anomalies}")

    if num_anomalies > 0:
        # Display enhanced anomalies table with highlighting
        st.subheader("Anomalies Detected")
        
        # Add legend for color coding
        st.markdown("""
        **Color Legend:**
        - 🔴 **Red with white text**: Primary anomaly contributor (highest deviation)
        - � **Orange with black text**: Secondary contributor  
        - 🔵 **Light blue with black text**: Minor contributor
        """)
        
        # Apply styling and display
        styled_anomalies = highlight_anomalous_values(anomalies, contributions, anomaly_indices)
        st.dataframe(styled_anomalies, width='stretch')
        
        # Show detailed anomaly insights
        with st.expander("📊 Detailed Anomaly Analysis"):
            max_details = min(max_anomaly_details, len(anomaly_indices))
            for i, idx in enumerate(anomaly_indices[:max_details]):
                st.write(f"**Row {idx} Anomaly Breakdown:**")
                row_contributions = contributions[idx]
                sorted_contribs = sorted(row_contributions.items(), key=lambda x: x[1], reverse=True)
                
                cols = st.columns(len(sorted_contribs[:3]))  # Show top 3 contributors
                for j, (feature, contrib) in enumerate(sorted_contribs[:3]):
                    with cols[j]:
                        value = df.iloc[idx][feature]
                        st.metric(
                            feature, 
                            f"{value:.2f}",
                            f"{contrib:.2f}σ deviation"
                        )
                if i < max_details - 1:
                    st.markdown("---")
    
    else:
        st.success("🎉 No anomalies detected in your data!")
        st.info("Your data appears to be within normal operating parameters.")
        st.markdown("*Try adjusting the contamination rate in the sidebar if you suspect there should be anomalies.*")

    # Generate and display enhanced graphs
    if num_anomalies > 0:
        st.subheader("📈 Anomaly Visualizations")
        
        # Create tabs for different visualization types
        tab1, tab2 = st.tabs(["Individual Features", "Feature Correlations"])
        
        with tab1:
            # Show individual feature plots
            for col in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot all data
                ax.plot(df.index, df[col], label="Normal Data", alpha=0.7, color='blue')
                
                # Highlight anomalies
                anomaly_values = df[col].iloc[anomaly_indices]
                ax.scatter(anomaly_indices, anomaly_values, color='red', s=100, 
                          label=f"Anomalies ({len(anomaly_indices)})", zorder=5)
                
                # Add statistics
                mean_val = df[col].mean()
                std_val = df[col].std()
                ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axhline(y=mean_val + 2*std_val, color='orange', linestyle=':', alpha=0.7, label='+2σ')
                ax.axhline(y=mean_val - 2*std_val, color='orange', linestyle=':', alpha=0.7, label='-2σ')
                
                ax.set_title(f"Anomaly Detection: {col}")
                ax.set_xlabel("Row Index")
                ax.set_ylabel(col)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            # Create correlation heatmap for anomalous rows
            if len(anomalies) > 1 and show_correlation_heatmap:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = anomalies.corr()
                
                if HAS_SEABORN:
                    # Use seaborn if available
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, ax=ax)
                else:
                    # Fallback to matplotlib
                    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                    
                    # Add text annotations
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(len(correlation_matrix.columns)):
                            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black")
                    
                    # Set tick labels
                    ax.set_xticks(range(len(correlation_matrix.columns)))
                    ax.set_yticks(range(len(correlation_matrix.columns)))
                    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                    ax.set_yticklabels(correlation_matrix.columns)
                
                ax.set_title("Feature Correlations in Anomalous Data")
                st.pyplot(fig)
                plt.close()
            elif not show_correlation_heatmap:
                st.info("Correlation heatmap disabled in settings.")
            else:
                st.info("Need at least 2 anomalies to show correlation analysis.")
    
    else:
        # Show normal data visualization when no anomalies found
        st.subheader("📊 Data Overview")
        
        for col in df.columns[:3]:  # Show first 3 columns to avoid clutter
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df[col], label=f"{col} Data", color='blue')
            
            mean_val = df[col].mean()
            ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            
            ax.set_title(f"Data Overview: {col}")
            ax.set_xlabel("Row Index") 
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()