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

def detect_transaction_date_columns(df):
    """
    Detect date columns that represent transaction timing events.
    Prioritizes columns with keywords like 'starts', 'accepted', 'created', etc.
    Returns list of dicts with column name, score, and matched keyword.
    """
    # Priority keywords (higher priority = more likely to be transaction time)
    priority_keywords = {
        'high': ['start', 'accept', 'create', 'complete', 'transaction', 'occur', 'place', 'process', 'submit'],
        'medium': ['date', 'time', 'timestamp', 'when'],
        'low': ['modif', 'update', 'change', 'edit']
    }
    
    # First, detect all date columns using existing function
    all_date_columns = detect_date_columns(df)
    
    if not all_date_columns:
        return []
    
    # Score each date column based on keywords in name
    scored_columns = []
    
    for col in all_date_columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        score = 0
        matched_keyword = None
        
        # Check high priority keywords (score = 3)
        for keyword in priority_keywords['high']:
            if keyword in col_lower:
                score = 3
                matched_keyword = keyword
                break
        
        # Check medium priority (score = 2)
        if score == 0:
            for keyword in priority_keywords['medium']:
                if keyword in col_lower:
                    score = 2
                    matched_keyword = keyword
                    break
        
        # Check low priority (score = 1)
        if score == 0:
            for keyword in priority_keywords['low']:
                if keyword in col_lower:
                    score = 1
                    matched_keyword = keyword
                    break
        
        # If no keywords matched, give it a neutral score
        if score == 0:
            score = 2
            matched_keyword = 'generic'
        
        scored_columns.append({
            'column': col,
            'score': score,
            'keyword': matched_keyword
        })
    
    # Sort by score (descending), then alphabetically
    scored_columns.sort(key=lambda x: (-x['score'], x['column']))
    
    return scored_columns

def detect_revenue_columns(df):
    """
    Auto-detect likely revenue/amount columns with priority scoring.
    Prioritizes columns with specific financial keywords.
    Returns list of dicts with column name, score, and matched keyword.
    """
    # Priority keywords (higher priority = more likely to be primary revenue column)
    priority_keywords = {
        'high': ['gpv', 'revenue', 'gross', 'net', 'income', 'earnings', 'profit'],
        'medium': ['amount', 'total', 'sales', 'price', 'value', 'payment', 'charge'],
        'low': ['fee', 'cost', 'rate', 'tax', 'discount', 'tip', 'commission']
    }
    
    scored_columns = []
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            score = 0
            matched_keyword = None
            
            # Check high priority keywords (score = 3)
            for keyword in priority_keywords['high']:
                if keyword in col_lower:
                    score = 3
                    matched_keyword = keyword
                    break
            
            # Check medium priority (score = 2)
            if score == 0:
                for keyword in priority_keywords['medium']:
                    if keyword in col_lower:
                        score = 2
                        matched_keyword = keyword
                        break
            
            # Check low priority (score = 1)
            if score == 0:
                for keyword in priority_keywords['low']:
                    if keyword in col_lower:
                        score = 1
                        matched_keyword = keyword
                        break
            
            # Only include columns that matched a keyword
            if score > 0:
                scored_columns.append({
                    'column': col,
                    'score': score,
                    'keyword': matched_keyword
                })
    
    # Sort by score (descending), then alphabetically
    scored_columns.sort(key=lambda x: (-x['score'], x['column']))
    
    return scored_columns

def get_revenue_column_names(df):
    """Helper function to return just column names for backwards compatibility"""
    scored = detect_revenue_columns(df)
    return [item['column'] for item in scored]

def detect_transaction_type_column(df):
    """
    Detect columns that likely contain transaction type information.
    Look for columns with keywords like 'type', 'category', 'product', 'plan'
    """
    type_keywords = ['type', 'category', 'product', 'plan', 'subscription', 'service', 'item']
    
    candidates = []
    for col in df.columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        if any(keyword in col_lower for keyword in type_keywords):
            # Verify it's a categorical/string column (object type)
            if df[col].dtype == 'object':
                candidates.append(col)
    
    return candidates

def filter_subscriptions(df, type_col, exclude_subscriptions=True):
    """
    Filter out subscription transactions from the dataframe.
    Matches 'subscription', 'subscriptions', 'Subscription(s)', etc.
    
    Returns tuple of (filtered_df, num_filtered)
    """
    if not exclude_subscriptions or type_col is None:
        return df, 0
    
    original_count = len(df)
    
    # Case-insensitive match for 'subscription' anywhere in the value
    mask = ~df[type_col].str.lower().str.contains('subscription', na=False)
    filtered_df = df[mask]
    
    num_filtered = original_count - len(filtered_df)
    
    return filtered_df, num_filtered

def detect_location_column(df):
    """
    Detect columns that likely contain location/site information.
    Look for columns with keywords like 'location', 'site', 'loc_id', 'store', 'branch'
    """
    location_keywords = ['location', 'site', 'loc_id', 'loc id', 'store', 'branch', 'outlet', 'venue', 'facility', 'terminal']
    
    candidates = []
    for col in df.columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        if any(keyword in col_lower for keyword in location_keywords):
            # Accept string columns or ID columns (which could be numeric)
            if df[col].dtype == 'object' or 'id' in col.lower():
                candidates.append(col)
    
    return candidates

def filter_by_location(df, location_col, selected_locations):
    """
    Filter dataframe to only include rows matching selected locations.
    
    Args:
        df: DataFrame to filter
        location_col: Name of the location column
        selected_locations: List of location values to include
    
    Returns tuple of (filtered_df, num_excluded)
    """
    if not selected_locations or location_col is None:
        return df, 0
    
    original_count = len(df)
    
    # Filter to only include selected locations
    mask = df[location_col].isin(selected_locations)
    filtered_df = df[mask]
    
    num_excluded = original_count - len(filtered_df)
    
    return filtered_df, num_excluded

def calculate_day_of_week_metrics(df, date_col, revenue_col=None):
    """Calculate transaction metrics grouped by day of week"""
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Convert date column to datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Remove rows with invalid dates
    df_copy = df_copy.dropna(subset=[date_col])
    
    if len(df_copy) == 0:
        return None
    
    # Extract day of week (0=Monday, 6=Sunday)
    df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek
    df_copy['day_name'] = df_copy[date_col].dt.day_name()
    
    # Count transactions by day of week
    transaction_counts = df_copy.groupby(['day_of_week', 'day_name']).size()
    transaction_counts.index = transaction_counts.index.get_level_values('day_name')
    
    # Reorder to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    transaction_counts = transaction_counts.reindex([day for day in day_order if day in transaction_counts.index])
    
    result = {
        'transaction_counts': transaction_counts,
        'revenue_totals': None,
        'average_rates': None
    }
    
    # Calculate revenue metrics if revenue column is provided and valid
    if revenue_col and revenue_col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[revenue_col]):
            # Group by day name for revenue calculations
            df_copy['day_name'] = df_copy[date_col].dt.day_name()
            
            # Calculate total revenue by day
            revenue_totals = df_copy.groupby('day_name')[revenue_col].sum()
            revenue_totals = revenue_totals.reindex([day for day in day_order if day in revenue_totals.index])
            
            # Calculate average rate (revenue per transaction)
            average_rates = revenue_totals / transaction_counts
            
            result['revenue_totals'] = revenue_totals
            result['average_rates'] = average_rates
    
    return result

def create_day_of_week_charts(dow_metrics, revenue_col_name=None):
    """Generate three visualization charts for day of week analysis"""
    
    if dow_metrics is None:
        st.warning("No valid day of week data available.")
        return
    
    transaction_counts = dow_metrics['transaction_counts']
    revenue_totals = dow_metrics['revenue_totals']
    average_rates = dow_metrics['average_rates']
    
    # Create three columns for the charts
    col1, col2, col3 = st.columns(3)
    
    # Chart 1: Transaction Volume by Day
    with col1:
        st.subheader("📊 Transaction Volume")
        
        if HAS_PLOTLY:
            # Color weekends differently
            colors = ['#1f77b4' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                     for day in transaction_counts.index]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=transaction_counts.index,
                    y=transaction_counts.values,
                    marker_color=colors,
                    text=transaction_counts.values,
                    textposition='outside'
                )
            ])
            
            # Add average line
            avg_transactions = transaction_counts.mean()
            fig.add_hline(y=avg_transactions, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_transactions:.0f}")
            
            fig.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Number of Transactions",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Matplotlib fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#1f77b4' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                     for day in transaction_counts.index]
            
            bars = ax.bar(transaction_counts.index, transaction_counts.values, color=colors)
            
            # Add average line
            avg_transactions = transaction_counts.mean()
            ax.axhline(y=avg_transactions, color='red', linestyle='--', alpha=0.7,
                      label=f'Avg: {avg_transactions:.0f}')
            
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("Number of Transactions")
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
    
    # Chart 2: Revenue by Day
    with col2:
        st.subheader("💰 Total Revenue")
        
        if revenue_totals is not None:
            if HAS_PLOTLY:
                colors = ['#2ca02c' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                         for day in revenue_totals.index]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=revenue_totals.index,
                        y=revenue_totals.values,
                        marker_color=colors,
                        text=[f'${v:,.0f}' for v in revenue_totals.values],
                        textposition='outside'
                    )
                ])
                
                # Add average line
                avg_revenue = revenue_totals.mean()
                fig.add_hline(y=avg_revenue, line_dash="dash", line_color="red",
                             annotation_text=f"Avg: ${avg_revenue:,.0f}")
                
                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Total Revenue ($)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Matplotlib fallback
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ca02c' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                         for day in revenue_totals.index]
                
                bars = ax.bar(revenue_totals.index, revenue_totals.values, color=colors)
                
                # Add average line
                avg_revenue = revenue_totals.mean()
                ax.axhline(y=avg_revenue, color='red', linestyle='--', alpha=0.7,
                          label=f'Avg: ${avg_revenue:,.0f}')
                
                ax.set_xlabel("Day of Week")
                ax.set_ylabel("Total Revenue ($)")
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No revenue column selected")
    
    # Chart 3: Average Rate by Day
    with col3:
        st.subheader("📈 Average Rate")
        
        if average_rates is not None:
            if HAS_PLOTLY:
                colors = ['#9467bd' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                         for day in average_rates.index]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=average_rates.index,
                        y=average_rates.values,
                        marker_color=colors,
                        text=[f'${v:.2f}' for v in average_rates.values],
                        textposition='outside'
                    )
                ])
                
                # Add average line
                avg_rate = average_rates.mean()
                fig.add_hline(y=avg_rate, line_dash="dash", line_color="red",
                             annotation_text=f"Avg: ${avg_rate:.2f}")
                
                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Revenue per Transaction ($)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Matplotlib fallback
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#9467bd' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                         for day in average_rates.index]
                
                bars = ax.bar(average_rates.index, average_rates.values, color=colors)
                
                # Add average line
                avg_rate = average_rates.mean()
                ax.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.7,
                          label=f'Avg: ${avg_rate:.2f}')
                
                ax.set_xlabel("Day of Week")
                ax.set_ylabel("Revenue per Transaction ($)")
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No revenue column selected")
    
    # Summary Statistics
    st.subheader("📊 Weekly Pattern Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        busiest_day = transaction_counts.idxmax()
        busiest_count = transaction_counts.max()
        st.metric("Busiest Day", busiest_day, f"{busiest_count:,} transactions")
    
    with col2:
        if revenue_totals is not None:
            highest_revenue_day = revenue_totals.idxmax()
            highest_revenue = revenue_totals.max()
            st.metric("Highest Revenue Day", highest_revenue_day, f"${highest_revenue:,.2f}")
        else:
            st.metric("Highest Revenue Day", "N/A", "No revenue data")
    
    with col3:
        if average_rates is not None:
            best_rate_day = average_rates.idxmax()
            best_rate = average_rates.max()
            st.metric("Best Rate Day", best_rate_day, f"${best_rate:.2f}/txn")
        else:
            st.metric("Best Rate Day", "N/A", "No revenue data")

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
    
    show_day_of_week_analysis = st.checkbox(
        "Show Day of Week Analysis",
        value=True,
        help="Analyze transaction patterns by day of week (requires date columns)"
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
    
    # Transaction Filters
    st.subheader("Transaction Filters")
    exclude_subscriptions = st.checkbox(
        "Exclude Subscription Transactions",
        value=True,
        help="Remove subscription-type transactions from daily volume and day-of-week analysis"
    )
    
    enable_location_filter = st.checkbox(
        "Filter by Location",
        value=False,
        help="Filter transactions to specific locations/sites"
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
        
        # Subscription filtering
        filtered_df_for_volume = original_df
        type_columns = detect_transaction_type_column(original_df)
        
        if exclude_subscriptions and type_columns:
            selected_type_col_volume = st.selectbox(
                "Transaction type column (for filtering):",
                type_columns,
                key="type_col_volume"
            )
            filtered_df_for_volume, num_filtered = filter_subscriptions(
                original_df, selected_type_col_volume, exclude_subscriptions
            )
            if num_filtered > 0:
                st.info(f"🔽 Filtered out **{num_filtered:,}** subscription transactions (showing {len(filtered_df_for_volume):,} of {len(original_df):,})")
        elif exclude_subscriptions and not type_columns:
            st.warning("⚠️ No transaction type column detected for subscription filtering. Showing all transactions.")
        
        # Location filtering
        if enable_location_filter:
            location_columns = detect_location_column(original_df)
            
            if location_columns:
                selected_location_col_volume = st.selectbox(
                    "Location column:",
                    location_columns,
                    key="location_col_volume"
                )
                
                # Get unique locations for multi-select
                unique_locations = filtered_df_for_volume[selected_location_col_volume].dropna().unique().tolist()
                # Sort locations (handle mixed types)
                try:
                    unique_locations = sorted(unique_locations, key=str)
                except:
                    unique_locations = sorted([str(loc) for loc in unique_locations])
                
                selected_locations_volume = st.multiselect(
                    "Select locations to include:",
                    unique_locations,
                    default=unique_locations,  # All selected by default
                    key="location_multiselect_volume"
                )
                
                if selected_locations_volume:
                    filtered_df_for_volume, num_excluded = filter_by_location(
                        filtered_df_for_volume, selected_location_col_volume, selected_locations_volume
                    )
                    st.info(f"📍 Showing **{len(selected_locations_volume)}** of {len(unique_locations)} locations ({len(filtered_df_for_volume):,} transactions)")
                else:
                    st.warning("⚠️ No locations selected. Please select at least one location.")
            else:
                st.warning("⚠️ No location column detected. Cannot filter by location.")
        
        # Analyze transaction patterns
        if selected_date_col:
            try:
                daily_counts = count_daily_transactions(filtered_df_for_volume, selected_date_col)
                create_transaction_summary_charts(daily_counts)
                
                # Add separator
                st.markdown("---")
                
            except Exception as e:
                st.warning(f"Could not analyze transaction patterns: {str(e)}")
    
    elif date_columns and not show_transaction_analysis:
        st.info("💡 Transaction analysis disabled in settings. Enable it in the sidebar to see transaction volume patterns.")
    else:
        st.info("💡 No date columns detected. Upload a file with date/time columns to see transaction volume analysis.")
    
    # Day of Week Analysis Section
    if date_columns and show_day_of_week_analysis:
        st.header("📅 Day of Week Analysis")
        
        # Detect revenue columns
        revenue_columns = detect_revenue_columns(original_df)
        
        # Get scored transaction date columns
        scored_date_columns = detect_transaction_date_columns(original_df)
        
        # Smart date column selection
        if scored_date_columns:
            # Auto-select the highest priority column
            default_col = scored_date_columns[0]['column']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                dow_date_col = st.selectbox(
                    "Select transaction date column:",
                    [item['column'] for item in scored_date_columns],
                    index=0,  # Default to highest priority
                    key="dow_date_selector"
                )
            with col2:
                # Show why this column was recommended
                selected_info = next(
                    (item for item in scored_date_columns 
                     if item['column'] == dow_date_col), 
                    None
                )
                if selected_info:
                    if selected_info['score'] == 3:
                        st.success("✓ High Priority")
                    elif selected_info['score'] == 2:
                        st.info("ℹ️ Medium Priority")
                    else:
                        st.warning("⚠️ Low Priority")
            
            # Show explanation of column selection
            with st.expander("ℹ️ About Date Column Selection"):
                st.markdown("""
                **Date columns are prioritized as follows:**
                - 🟢 **High Priority**: Columns containing 'starts', 'accepted', 'created', 
                  'completed', 'transaction', 'placed' - best represent when transactions occur
                - 🟡 **Medium Priority**: Generic date/time columns
                - 🟠 **Low Priority**: Modified/updated columns - may not reflect original 
                  transaction time
                """)
                
                # Show all detected columns with scores
                st.write("**Detected Date Columns:**")
                for item in scored_date_columns:
                    if item['score'] == 3:
                        priority = "🟢 High"
                    elif item['score'] == 2:
                        priority = "🟡 Medium"
                    else:
                        priority = "🟠 Low"
                    
                    keyword_info = f" (contains '{item['keyword']}')" if item['keyword'] != 'generic' else ""
                    st.write(f"- **{item['column']}**: {priority}{keyword_info}")
        else:
            # Fallback if no date columns detected (shouldn't happen due to parent if condition)
            dow_date_col = date_columns[0] if date_columns else None
            st.warning("Could not analyze date column priorities.")
        
        # Revenue column selection with smart detection
        selected_revenue_col = None
        scored_revenue_columns = revenue_columns  # Already scored from detect_revenue_columns
        
        if scored_revenue_columns:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Build options list with "None" first, then scored columns
                revenue_options = ["None"] + [item['column'] for item in scored_revenue_columns]
                selected_revenue_col = st.selectbox(
                    "Select revenue column:",
                    revenue_options,
                    index=1 if len(revenue_options) > 1 else 0,  # Default to first detected column
                    key="revenue_selector"
                )
                if selected_revenue_col == "None":
                    selected_revenue_col = None
            
            with col2:
                # Show priority indicator for selected column
                if selected_revenue_col:
                    selected_rev_info = next(
                        (item for item in scored_revenue_columns 
                         if item['column'] == selected_revenue_col), 
                        None
                    )
                    if selected_rev_info:
                        if selected_rev_info['score'] == 3:
                            st.success("✓ High Priority")
                        elif selected_rev_info['score'] == 2:
                            st.info("ℹ️ Medium Priority")
                        else:
                            st.warning("⚠️ Low Priority")
                else:
                    st.info("💡 Optional")
            
            # Show explanation of revenue column selection
            with st.expander("ℹ️ About Revenue Column Selection"):
                st.markdown("""
                **Revenue columns are prioritized as follows:**
                - 🟢 **High Priority**: Columns containing 'gpv', 'revenue', 'gross', 'net', 
                  'income', 'earnings', 'profit' - primary financial metrics
                - 🟡 **Medium Priority**: 'amount', 'total', 'sales', 'price', 'value', 'payment'
                - 🟠 **Low Priority**: 'fee', 'cost', 'rate', 'tax', 'discount', 'tip', 'commission'
                """)
                
                # Show all detected columns with scores
                st.write("**Detected Revenue Columns:**")
                for item in scored_revenue_columns:
                    if item['score'] == 3:
                        priority = "🟢 High"
                    elif item['score'] == 2:
                        priority = "🟡 Medium"
                    else:
                        priority = "🟠 Low"
                    
                    st.write(f"- **{item['column']}**: {priority} (contains '{item['keyword']}')")
        else:
            st.info("💡 No revenue columns detected. Analysis will show transaction counts only.")
        
        # Subscription filtering for day of week analysis
        filtered_df_for_dow = original_df
        type_columns_dow = detect_transaction_type_column(original_df)
        
        if exclude_subscriptions and type_columns_dow:
            # Use the same type column if already selected in Transaction Volume, otherwise let user select
            if 'selected_type_col_volume' in locals() and selected_type_col_volume in type_columns_dow:
                selected_type_col_dow = selected_type_col_volume
                st.caption(f"Using transaction type column: **{selected_type_col_dow}**")
            else:
                selected_type_col_dow = st.selectbox(
                    "Transaction type column (for filtering):",
                    type_columns_dow,
                    key="type_col_dow"
                )
            
            filtered_df_for_dow, num_filtered_dow = filter_subscriptions(
                original_df, selected_type_col_dow, exclude_subscriptions
            )
            if num_filtered_dow > 0:
                st.info(f"🔽 Filtered out **{num_filtered_dow:,}** subscription transactions (showing {len(filtered_df_for_dow):,} of {len(original_df):,})")
        elif exclude_subscriptions and not type_columns_dow:
            st.warning("⚠️ No transaction type column detected for subscription filtering. Showing all transactions.")
        
        # Location filtering for day of week analysis
        if enable_location_filter:
            location_columns_dow = detect_location_column(original_df)
            
            if location_columns_dow:
                # Use the same location column if already selected in Transaction Volume
                if 'selected_location_col_volume' in locals() and selected_location_col_volume in location_columns_dow:
                    selected_location_col_dow = selected_location_col_volume
                    st.caption(f"Using location column: **{selected_location_col_dow}**")
                else:
                    selected_location_col_dow = st.selectbox(
                        "Location column:",
                        location_columns_dow,
                        key="location_col_dow"
                    )
                
                # Get unique locations for multi-select
                unique_locations_dow = filtered_df_for_dow[selected_location_col_dow].dropna().unique().tolist()
                try:
                    unique_locations_dow = sorted(unique_locations_dow, key=str)
                except:
                    unique_locations_dow = sorted([str(loc) for loc in unique_locations_dow])
                
                # Use same selections if available from Transaction Volume
                if 'selected_locations_volume' in locals() and selected_locations_volume:
                    default_locations_dow = [loc for loc in selected_locations_volume if loc in unique_locations_dow]
                else:
                    default_locations_dow = unique_locations_dow
                
                selected_locations_dow = st.multiselect(
                    "Select locations to include:",
                    unique_locations_dow,
                    default=default_locations_dow,
                    key="location_multiselect_dow"
                )
                
                if selected_locations_dow:
                    filtered_df_for_dow, num_excluded_dow = filter_by_location(
                        filtered_df_for_dow, selected_location_col_dow, selected_locations_dow
                    )
                    st.info(f"📍 Showing **{len(selected_locations_dow)}** of {len(unique_locations_dow)} locations ({len(filtered_df_for_dow):,} transactions)")
                else:
                    st.warning("⚠️ No locations selected. Please select at least one location.")
            else:
                st.warning("⚠️ No location column detected. Cannot filter by location.")
        
        # Calculate and display day of week metrics
        try:
            dow_metrics = calculate_day_of_week_metrics(
                filtered_df_for_dow, 
                dow_date_col, 
                selected_revenue_col
            )
            
            if dow_metrics:
                create_day_of_week_charts(dow_metrics, selected_revenue_col)
            else:
                st.warning("Could not calculate day of week metrics. Please check your date column.")
        
        except Exception as e:
            st.error(f"Error in day of week analysis: {str(e)}")
        
        # Add separator
        st.markdown("---")
    
    elif date_columns and not show_day_of_week_analysis:
        st.info("💡 Day of week analysis disabled in settings. Enable it in the sidebar to see weekly patterns.")
    
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