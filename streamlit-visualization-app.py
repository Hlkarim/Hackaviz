import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Data Visualization Dashboard")
st.markdown("This dashboard showcases various data visualization techniques using sample datasets.")

# Function to generate sample data
@st.cache_data
def generate_sample_data():
    # Time series data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(loc=100, scale=15, size=len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 30,
        'visitors': np.random.normal(loc=500, scale=50, size=len(dates)) + np.cos(np.arange(len(dates)) * 0.1) * 100
    })
    
    # Categorical data
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home Goods']
    cat_data = pd.DataFrame({
        'category': categories,
        'sales': np.random.randint(50000, 200000, len(categories)),
        'profit': np.random.randint(5000, 50000, len(categories)),
        'growth': np.random.uniform(-0.2, 0.5, len(categories))
    })
    
    # Correlation data
    n_samples = 200
    corr_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples)
    })
    # Add some correlations
    corr_data['feature_2'] = corr_data['feature_1'] * 0.8 + corr_data['feature_2'] * 0.2
    corr_data['feature_4'] = corr_data['feature_3'] * -0.6 + corr_data['feature_4'] * 0.4
    
    # Geographic data (US states)
    state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                   'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                   'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                   'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                   'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    geo_data = pd.DataFrame({
        'state': state_codes,
        'value': np.random.randint(10, 100, len(state_codes))
    })
    
    return ts_data, cat_data, corr_data, geo_data

# Generate sample data
ts_data, cat_data, corr_data, geo_data = generate_sample_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a visualization category",
    ["Time Series", "Bar & Pie Charts", "Distributions", "Correlations", "Geographical"]
)

# Time Series Visualizations
if page == "Time Series":
    st.header("Time Series Visualizations")
    
    # Allow user to select date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", ts_data['date'].min().date())
    with col2:
        end_date = st.date_input("End Date", ts_data['date'].max().date())
    
    # Filter data based on date range
    filtered_ts = ts_data[(ts_data['date'].dt.date >= start_date) & (ts_data['date'].dt.date <= end_date)]
    
    # Time series plot tabs
    ts_tab1, ts_tab2, ts_tab3 = st.tabs(["Line Chart", "Area Chart", "Moving Averages"])
    
    with ts_tab1:
        st.subheader("Line Chart")
        fig = px.line(filtered_ts, x='date', y=['sales', 'visitors'], 
                      title="Sales and Visitors Over Time",
                      labels={'value': 'Count', 'date': 'Date', 'variable': 'Metric'},
                      template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with ts_tab2:
        st.subheader("Area Chart")
        fig = px.area(filtered_ts, x='date', y='sales',
                      title="Sales Volume Over Time",
                      labels={'sales': 'Sales', 'date': 'Date'},
                      template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with ts_tab3:
        st.subheader("Moving Averages")
        # Calculate moving averages
        ma_window = st.slider("Moving Average Window (days)", min_value=3, max_value=30, value=7)
        filtered_ts['sales_ma'] = filtered_ts['sales'].rolling(window=ma_window).mean()
        
        fig = px.line(filtered_ts, x='date', y=['sales', 'sales_ma'], 
                      title=f"Sales with {ma_window}-day Moving Average",
                      labels={'value': 'Sales', 'date': 'Date', 'variable': 'Series'},
                      template="plotly_white")
        fig.update_traces(line=dict(width=1), selector=dict(name='sales'))
        fig.update_traces(line=dict(width=3), selector=dict(name='sales_ma'))
        st.plotly_chart(fig, use_container_width=True)

# Bar & Pie Charts
elif page == "Bar & Pie Charts":
    st.header("Bar & Pie Charts")
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Bar Chart", "Grouped Bar Chart", "Pie Chart"])
    
    with chart_tab1:
        st.subheader("Bar Chart")
        fig = px.bar(cat_data, x='category', y='sales', 
                    title="Sales by Category",
                    color='sales',
                    labels={'sales': 'Sales ($)', 'category': 'Product Category'},
                    template="plotly_white",
                    color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_tab2:
        st.subheader("Grouped Bar Chart")
        # Prepare data for grouped bar chart
        melted_data = pd.melt(cat_data, id_vars=['category'], value_vars=['sales', 'profit'],
                            var_name='metric', value_name='value')
        
        fig = px.bar(melted_data, x='category', y='value', color='metric',
                    title="Sales and Profit by Category",
                    labels={'value': 'Amount ($)', 'category': 'Product Category', 'metric': 'Metric'},
                    template="plotly_white",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_tab3:
        st.subheader("Pie Chart")
        fig = px.pie(cat_data, values='sales', names='category',
                    title="Sales Distribution by Category",
                    template="plotly_white",
                    hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# Distribution Visualizations
elif page == "Distributions":
    st.header("Distribution Visualizations")
    
    dist_tab1, dist_tab2 = st.tabs(["Histogram", "Box Plot"])
    
    with dist_tab1:
        st.subheader("Histogram")
        
        # Generate some normally distributed data
        hist_data = np.random.normal(loc=50, scale=15, size=1000)
        hist_df = pd.DataFrame({'values': hist_data})
        
        bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
        
        fig = px.histogram(hist_df, x='values', nbins=bins,
                        title="Histogram of Values",
                        labels={'values': 'Value', 'count': 'Frequency'},
                        template="plotly_white")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_tab2:
        st.subheader("Box Plot")
        
        # Generate data for box plot
        categories = ['Group A', 'Group B', 'Group C', 'Group D']
        box_data = []
        
        for cat in categories:
            if cat == 'Group A':
                values = np.random.normal(loc=50, scale=10, size=100)
            elif cat == 'Group B':
                values = np.random.normal(loc=60, scale=15, size=100)
            elif cat == 'Group C':
                values = np.random.normal(loc=45, scale=5, size=100)
            else:
                values = np.random.normal(loc=55, scale=20, size=100)
            
            for val in values:
                box_data.append({'group': cat, 'value': val})
        
        box_df = pd.DataFrame(box_data)
        
        fig = px.box(box_df, x='group', y='value',
                    title="Box Plot by Group",
                    labels={'group': 'Group', 'value': 'Value'},
                    template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Correlation Visualizations
elif page == "Correlations":
    st.header("Correlation Visualizations")
    
    corr_tab1, corr_tab2, corr_tab3 = st.tabs(["Correlation Matrix", "Scatter Plot", "Pair Plot"])
    
    with corr_tab1:
        st.subheader("Correlation Matrix")
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix, 
                        text_auto='.2f',
                        aspect="equal",
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with corr_tab2:
        st.subheader("Scatter Plot")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature", corr_data.columns.tolist(), index=0)
        with col2:
            y_feature = st.selectbox("Y-axis feature", corr_data.columns.tolist(), index=1)
        
        fig = px.scatter(corr_data, x=x_feature, y=y_feature,
                        title=f"Scatter Plot: {x_feature} vs {y_feature}",
                        labels={x_feature: x_feature, y_feature: y_feature},
                        template="plotly_white",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display correlation coefficient
        corr_coef = corr_data[x_feature].corr(corr_data[y_feature])
        st.info(f"Correlation coefficient between {x_feature} and {y_feature}: {corr_coef:.4f}")
    
    with corr_tab3:
        st.subheader("Pair Plot")
        st.warning("Generating a full pair plot may be performance-intensive. Select fewer features for better performance.")
        
        selected_features = st.multiselect(
            "Select features for pair plot (2-5 recommended)",
            options=corr_data.columns.tolist(),
            default=corr_data.columns.tolist()[:3]
        )
        
        if len(selected_features) >= 2:
            # Using matplotlib/seaborn for pair plot
            fig, ax = plt.subplots(figsize=(10, 8))
            pair_plot = sns.pairplot(corr_data[selected_features], diag_kind='kde', plot_kws={'alpha': 0.6})
            plt.tight_layout()
            st.pyplot(pair_plot.fig)
        else:
            st.error("Please select at least 2 features for the pair plot.")

# Geographical Visualizations
elif page == "Geographical":
    st.header("Geographical Visualizations")
    
    st.subheader("US States Choropleth Map")
    
    # Create choropleth map of US states
    fig = px.choropleth(geo_data, 
                        locations='state', 
                        locationmode="USA-states", 
                        color='value',
                        scope="usa",
                        color_continuous_scale="Viridis",
                        labels={'value': 'Value'},
                        title="Values by US State")
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a simple data table for reference
    st.subheader("Data Table")
    st.dataframe(geo_data.sort_values('value', ascending=False), use_container_width=True)

# Add information section at the bottom
st.markdown("---")
st.markdown("""
### About this Dashboard
This is a sample Streamlit data visualization dashboard that demonstrates various charting and visualization capabilities.
The data used is randomly generated for demonstration purposes.

### Features:
- Time series analysis with interactive date filtering
- Bar charts and pie charts for categorical data
- Distribution analysis with histograms and box plots
- Correlation analysis with correlation matrices and scatter plots
- Geographic data visualization with choropleth maps

To use this with your own data, you would replace the sample data generation functions with your data loading code.
""")

# Add a sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Information")
st.sidebar.markdown("All datasets are randomly generated for demonstration purposes.")
st.sidebar.markdown("Adjust parameters and explore different visualization options using the controls above.")
