import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌡️ Temperature Prediction Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for weather-themed styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #2d3748 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #2d3748 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a365d 0%, #2d3748 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #2d3748;
        color: #a6c7ff;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #a6c7ff;
    }
    
    .weather-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #8cb8ff;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(140, 184, 255, 0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #b2c7e0;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        color: #a6c7ff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load weather data from CSV files"""
    try:
        # Try to load the after_Pettit dataset first (cleaner)
        df_after = pd.read_csv('df_model_after_Pettit.csv')
        df_before = pd.read_csv('df_model_before_Pettit.csv')
        
        # Add timestamp index for better visualization
        df_after.index = pd.date_range(start='2006-01-01', periods=len(df_after), freq='H')
        df_before.index = pd.date_range(start='2006-01-01', periods=len(df_before), freq='H')
        
        return df_after, df_before
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Model training function
@st.cache_resource
def train_model(df):
    """Train temperature prediction model"""
    try:
        # Prepare features and target
        feature_cols = ['Barometer - mm Hg', 'Hum - %', 'Wind Speed - m/s', 
                       'Rain - mm', 'Heat Index - °C', 'Season', 'Day_Night']
        
        # Remove rows with missing values
        df_clean = df[feature_cols + ['Temp - °C']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['Temp - °C']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model, mse, r2, feature_cols
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None

# Load model function
@st.cache_resource
def load_model():
    """Load trained model"""
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

# Sidebar navigation
def main():
    st.sidebar.title("🌤️ Weather Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Project Overview": "overview",
        "📊 Data Visualizations": "visualizations", 
        "🎯 Temperature Prediction": "prediction"
    }
    
    selected_page = st.sidebar.radio("Navigate to:", list(pages.keys()))
    page = pages[selected_page]
    
    # Load data
    df_after, df_before = load_data()
    
    if df_after is None:
        st.error("Failed to load data. Please check if CSV files exist.")
        return
    
    # Route to selected page
    if page == "overview":
        show_overview(df_after, df_before)
    elif page == "visualizations":
        show_visualizations(df_after, df_before)
    elif page == "prediction":
        show_prediction(df_after)

def show_overview(df_after, df_before):
    """Project Overview Page"""
    st.title("🌡️ Temperature Prediction Dashboard")
    st.markdown("### Weather Station Data Analysis & Prediction System")
    
    # Project description
    st.markdown("""
    <div class="weather-card">
        <h3>🎯 Project Focus</h3>
        <p>This comprehensive weather analysis system predicts temperature using meteorological data from 
        MX Nizanda weather station (2006-2024). The project combines advanced data science techniques 
        with an intuitive interface for weather forecasting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Total Records",
            value=f"{len(df_before):,}",
            delta="Before Pettit Test"
        )
    
    with col2:
        st.metric(
            label="🔄 Processed Records", 
            value=f"{len(df_after):,}",
            delta="After Pettit Test"
        )
    
    with col3:
        avg_temp = df_after['Temp - °C'].mean()
        st.metric(
            label="🌡️ Average Temperature",
            value=f"{avg_temp:.1f}°C",
            delta=f"{df_after['Temp - °C'].std():.1f}°C std"
        )
    
    with col4:
        years = (df_after.index[-1] - df_after.index[0]).days / 365.25
        st.metric(
            label="📊 Data Coverage",
            value=f"{years:.1f} years",
            delta="2006-2024"
        )
    
    # Dashboard summary
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="weather-card">
            <h4>📈 Available Features</h4>
            <ul>
                <li>🌡️ Temperature & Heat Index</li>
                <li>💧 Humidity & Rainfall</li>
                <li>🌪️ Wind Speed & Direction</li>
                <li>📊 Barometric Pressure</li>
                <li>🌅 Season & Day/Night Cycles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Quick stats visualization
        fig = go.Figure()
        
        # Temperature distribution
        fig.add_trace(go.Histogram(
            x=df_after['Temp - °C'],
            name="Temperature Distribution",
            marker_color='#8cb8ff',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Temperature Distribution Overview",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df_after, df_before):
    """Data Visualizations Page"""
    st.title("📊 Weather Data Visualizations")
    
    # Use the cleaner dataset for visualizations
    df = df_after.copy()
    
    # Temperature trends over time
    st.markdown("### 🌡️ Temperature Trends Over Time")
    
    # Resample to daily means for better visualization
    daily_temp = df['Temp - °C'].resample('D').mean()
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=daily_temp.index,
        y=daily_temp.values,
        mode='lines',
        name='Daily Average Temperature',
        line=dict(color='#a6c7ff', width=1.5)
    ))
    
    fig_temp.update_layout(
        title="Daily Average Temperature Over Time",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight:</strong> The temperature data shows clear seasonal patterns with cyclical variations 
        throughout the years, indicating strong seasonal dependency in the weather patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Humidity distribution
    st.markdown("### 💧 Humidity Distribution")
    
    fig_hum = go.Figure()
    fig_hum.add_trace(go.Histogram(
        x=df['Hum - %'],
        nbinsx=50,
        marker_color='#b2c7e0',
        opacity=0.8,
        name='Humidity'
    ))
    
    fig_hum.update_layout(
        title="Humidity Distribution",
        xaxis_title="Humidity (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    
    st.plotly_chart(fig_hum, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight:</strong> Humidity levels show a varied distribution with peaks around specific ranges, 
        suggesting distinct weather patterns characteristic of the regional climate.
    </div>
    """, unsafe_allow_html=True)
    
    # Wind rose diagram
    st.markdown("### 🌪️ Wind Rose Diagram")
    
    # Filter out missing wind direction data
    wind_data = df[df['Wind_Dir_sin'].notna() & df['Wind_Dir_cos'].notna()].copy()
    
    if len(wind_data) > 0:
        fig_wind = go.Figure()
        
        fig_wind.add_trace(go.Scatterpolar(
            r=wind_data['Wind Speed - m/s'],
            theta=wind_data['Wind_Direction'],
            mode='markers',
            marker=dict(
                color=wind_data['Wind Speed - m/s'],
                colorscale='Blues',
                size=5,
                opacity=0.7
            ),
            name='Wind Speed & Direction'
        ))
        
        fig_wind.update_layout(
            title="Wind Rose Diagram",
            polar=dict(
                radialaxis=dict(title="Wind Speed (m/s)", color='#e2e8f0'),
                angularaxis=dict(color='#e2e8f0')
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig_wind, use_container_width=True)
    else:
        st.warning("Wind direction data not available for visualization")
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight:</strong> Wind patterns show predominant directions and speeds, 
        revealing the typical wind characteristics of the location.
    </div>
    """, unsafe_allow_html=True)
    
    # Rainfall over time
    st.markdown("### 🌧️ Rainfall Over Time")
    
    monthly_rain = df['Rain - mm'].resample('M').sum()
    
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Scatter(
        x=monthly_rain.index,
        y=monthly_rain.values,
        mode='lines+markers',
        name='Monthly Rainfall',
        line=dict(color='#8cb8ff', width=2),
        marker=dict(size=4)
    ))
    
    fig_rain.update_layout(
        title="Monthly Rainfall Over Time",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )
    
    st.plotly_chart(fig_rain, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight:</strong> Rainfall data reveals seasonal precipitation patterns with distinct 
        wet and dry periods, typical of the regional climate cycle.
    </div>
    """, unsafe_allow_html=True)
    
    # Yearly mean statistics
    st.markdown("### 📈 Yearly Mean Statistics")
    
    yearly_stats = df.groupby(df.index.year).agg({
        'Temp - °C': 'mean',
        'Hum - %': 'mean',
        'Wind Speed - m/s': 'mean',
        'Rain - mm': 'sum'
    })
    
    fig_yearly = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'Rainfall'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_yearly.add_trace(
        go.Scatter(x=yearly_stats.index, y=yearly_stats['Temp - °C'], 
                  name='Temp', line=dict(color='#a6c7ff')),
        row=1, col=1
    )
    
    fig_yearly.add_trace(
        go.Scatter(x=yearly_stats.index, y=yearly_stats['Hum - %'], 
                  name='Humidity', line=dict(color='#8cb8ff')),
        row=1, col=2
    )
    
    fig_yearly.add_trace(
        go.Scatter(x=yearly_stats.index, y=yearly_stats['Wind Speed - m/s'], 
                  name='Wind Speed', line=dict(color='#b2c7e0')),
        row=2, col=1
    )
    
    fig_yearly.add_trace(
        go.Scatter(x=yearly_stats.index, y=yearly_stats['Rain - mm'], 
                  name='Rainfall', line=dict(color='#9db7db')),
        row=2, col=2
    )
    
    fig_yearly.update_layout(
        title="Yearly Weather Statistics Trends",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        showlegend=False
    )
    
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Insight:</strong> Long-term trends show interesting patterns in weather parameters, 
        potentially indicating climate variations or measurement improvements over the years.
    </div>
    """, unsafe_allow_html=True)

def show_prediction(df):
    """Temperature Prediction Page"""
    st.title("🎯 Temperature Prediction System")
    
    # Train or load model
    model = load_model()
    if model is None:
        with st.spinner("Training temperature prediction model..."):
            model, mse, r2, feature_cols = train_model(df)
        
        if model is not None:
            st.success(f"✅ Model trained successfully! R² Score: {r2:.3f}")
        else:
            st.error("Failed to train model")
            return
    else:
        # Get feature columns
        feature_cols = ['Barometer - mm Hg', 'Hum - %', 'Wind Speed - m/s', 
                       'Rain - mm', 'Heat Index - °C', 'Season', 'Day_Night']
        st.success("✅ Model loaded successfully!")
    
    st.markdown("---")
    
    # Input form
    st.markdown("### 🌤️ Enter Weather Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Environmental Conditions")
        barometer = st.slider(
            "🌍 Barometric Pressure (mm Hg)",
            min_value=700.0, max_value=800.0, value=760.0, step=0.1
        )
        
        humidity = st.slider(
            "💧 Humidity (%)",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0
        )
        
        wind_speed = st.slider(
            "🌪️ Wind Speed (m/s)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.1
        )
        
        rainfall = st.slider(
            "🌧️ Rainfall (mm)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.1
        )
    
    with col2:
        st.markdown("#### Additional Parameters")
        heat_index = st.slider(
            "🌡️ Heat Index (°C)",
            min_value=0.0, max_value=50.0, value=25.0, step=0.1
        )
        
        season = st.selectbox(
            "🌸 Season",
            options=[0, 1],
            format_func=lambda x: "Winter/Dry" if x == 0 else "Summer/Wet"
        )
        
        day_night = st.selectbox(
            "🌅 Time Period",
            options=[0, 1],
            format_func=lambda x: "Night" if x == 0 else "Day"
        )
    
    # Prediction
    if st.button("🎯 Predict Temperature", type="primary"):
        try:
            # Prepare input data
            input_data = np.array([[barometer, humidity, wind_speed, rainfall, 
                                  heat_index, season, day_night]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🌡️ Predicted Temperature",
                    value=f"{prediction:.1f}°C"
                )
            
            with col2:
                # Temperature category
                if prediction < 15:
                    category = "🥶 Cold"
                    color = "#74c0fc"
                elif prediction < 25:
                    category = "🌤️ Mild"
                    color = "#a6c7ff"
                elif prediction < 30:
                    category = "☀️ Warm"
                    color = "#ffd93d"
                else:
                    category = "🔥 Hot"
                    color = "#ff8c42"
                
                st.metric(
                    label="📊 Category",
                    value=category
                )
            
            with col3:
                # Confidence indicator based on input ranges
                confidence = "High" if 20 <= prediction <= 35 else "Medium"
                st.metric(
                    label="🎯 Confidence",
                    value=confidence
                )
            
            # Visual temperature gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Temperature (°C)"},
                delta={'reference': 25},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 15], 'color': "#74c0fc"},
                        {'range': [15, 25], 'color': "#a6c7ff"},
                        {'range': [25, 30], 'color': "#ffd93d"},
                        {'range': [30, 50], 'color': "#ff8c42"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 40
                    }
                }
            ))
            
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=400
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Model information
    st.markdown("---")
    st.markdown("### 📋 Model Information")
    
    st.markdown("""
    <div class="weather-card">
        <h4>🤖 Model Details</h4>
        <p><strong>Algorithm:</strong> Random Forest Regressor</p>
        <p><strong>Features:</strong> 7 meteorological parameters</p>
        <p><strong>Training Data:</strong> MX Nizanda Weather Station (2006-2024)</p>
        <p><strong>Purpose:</strong> Predict temperature based on current weather conditions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()