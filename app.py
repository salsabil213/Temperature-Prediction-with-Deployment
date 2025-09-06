import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- THEME COLORS ---
BG_COLOR = "#151a28"
PLOT_BG = "#1c2234"
PANEL_BG = "#212a40"
FONT_COLOR = "#b2c7e0"
ACCENT_COLOR = "#8cb8ff"

st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌧️"
)

def set_custom_style():
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: {BG_COLOR};
            color: {FONT_COLOR};
        }}
        .sidebar .sidebar-content {{
            background: {PANEL_BG};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {ACCENT_COLOR};
            font-family: 'Arial Black', 'Segoe UI', sans-serif;
        }}
        .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: {BG_COLOR};
            border-radius: 8px;
            border: 2px solid {ACCENT_COLOR};
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0px 2px 10px #26344d33;
        }}
        .stSuccess, .stError {{
            padding: 18px 24px;
            border-radius: 12px;
            background: linear-gradient(90deg, #1c2234 60%, #212a40 100%);
            color: {ACCENT_COLOR};
            font-size: 22px;
            font-family: 'Segoe UI', 'Arial Black', sans-serif;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            border: 2px solid {ACCENT_COLOR};
            box-shadow: 0px 4px 20px #12162044;
        }}
        thead tr th {{
            background: {PANEL_BG};
            color: {ACCENT_COLOR};
        }}
        tbody tr td {{
            background: {BG_COLOR};
            color: {FONT_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# Sidebar navigation
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=80
)
st.sidebar.title("Weather Forecasting")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview & Data Stats",
        "Visualizations",
        "Model Deployment"
    ],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("#### Weather Forecasting Dashboard\nModern, interactive, and themed for meteorology 🌦️")

# --- DATA PATHS ---
VISUAL_DATA_PATH = "C:\\Users\\salsa\\OneDrive\\Desktop\\Python\\.venv\\AI Training\\cleaned_weather_data.csv"
MODEL_DATA_PATH = "C:\\Users\\salsa\\OneDrive\\Desktop\\Python\\.venv\\AI Training\\df_model_after_Pettit.csv"
MODEL_PATH = "C:\\Users\\salsa\\OneDrive\\Desktop\\Python\\.venv\\AI Training\\model.pkl"

# --- Wind direction mapping for model features ---
direction_map = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}
wind_dir_options = list(direction_map.keys())
wind_dir_label_map = {k: i for i, k in enumerate(wind_dir_options)}

# --- DATA LOADING ---
if not os.path.exists(VISUAL_DATA_PATH):
    st.error(f"Visualization data file not found: {VISUAL_DATA_PATH}. Please upload the file to the app directory.")
    st.stop()
if not os.path.exists(MODEL_DATA_PATH):
    st.error(f"Model data file not found: {MODEL_DATA_PATH}. Please upload the file to the app directory.")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please upload the file to the app directory.")
    st.stop()

@st.cache_data
def load_visual_data():
    df = pd.read_csv(VISUAL_DATA_PATH, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    return df

@st.cache_data
def load_model_data():
    df = pd.read_csv(MODEL_DATA_PATH)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df_visual = load_visual_data()
df_model = load_model_data()
model = load_model()

# ---- PAGE 1: OVERVIEW ----
if page == "Overview & Data Stats":
    st.markdown(
        f"""
        <div style="background:{PANEL_BG}; padding:28px; border-radius:16px; margin-bottom:24px;">
            <div style="display:flex; align-items:center;">
                <img src="https://images.pexels.com/photos/531756/pexels-photo-531756.jpeg?auto=compress&w=800" width="120" style="margin-right:32px; border-radius:18px; box-shadow:0px 4px 20px #12162044;">
                <div>
                    <h2 style="color:{ACCENT_COLOR}; margin-bottom:10px;">Weather Forecasting Project</h2>
                    <p style="font-size:20px;">A modern dashboard that visualizes and predicts weather conditions using advanced statistics and machine learning. Dive into trends of rainfall, temperature, and humidity, and explore the story your weather data tells!</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Key Weather Statistics")
    rain_mean = df_visual['Rain - mm'].mean() if 'Rain - mm' in df_visual.columns else None
    rain_max = df_visual['Rain - mm'].max() if 'Rain - mm' in df_visual.columns else None
    rain_min = df_visual['Rain - mm'].min() if 'Rain - mm' in df_visual.columns else None
    temp_mean = df_visual['Temp - °C'].mean() if 'Temp - °C' in df_visual.columns else None
    temp_max = df_visual['Temp - °C'].max() if 'Temp - °C' in df_visual.columns else None
    temp_min = df_visual['Temp - °C'].min() if 'Temp - °C' in df_visual.columns else None
    hum_mean = df_visual['Hum - %'].mean() if 'Hum - %' in df_visual.columns else None
    hum_max = df_visual['Hum - %'].max() if 'Hum - %' in df_visual.columns else None
    hum_min = df_visual['Hum - %'].min() if 'Hum - %' in df_visual.columns else None

    def fmt(val):
        return f"{val:.2f}" if val is not None else "N/A"

    st.markdown(
        f"""
        <div style="display: flex; gap: 28px;">
            <div style="background: {PLOT_BG}; padding:24px; border-radius:18px; flex:1;">
                <h4 style="color:{ACCENT_COLOR};">Rainfall (mm)</h4>
                <div style="font-size:36px; color:{ACCENT_COLOR}; font-weight:bold;">{fmt(rain_mean)}</div>
                <div style="font-size:18px;">Max: <span style="color:{ACCENT_COLOR};">{fmt(rain_max)}</span> &nbsp; | &nbsp; Min: <span style="color:{ACCENT_COLOR};">{fmt(rain_min)}</span></div>
            </div>
            <div style="background: {PLOT_BG}; padding:24px; border-radius:18px; flex:1;">
                <h4 style="color:{ACCENT_COLOR};">Temperature (°C)</h4>
                <div style="font-size:36px; color:{ACCENT_COLOR}; font-weight:bold;">{fmt(temp_mean)}</div>
                <div style="font-size:18px;">Max: <span style="color:{ACCENT_COLOR};">{fmt(temp_max)}</span> &nbsp; | &nbsp; Min: <span style="color:{ACCENT_COLOR};">{fmt(temp_min)}</span></div>
            </div>
            <div style="background: {PLOT_BG}; padding:24px; border-radius:18px; flex:1;">
                <h4 style="color:{ACCENT_COLOR};">Humidity (%)</h4>
                <div style="font-size:36px; color:{ACCENT_COLOR}; font-weight:bold;">{fmt(hum_mean)}</div>
                <div style="font-size:18px;">Max: <span style="color:{ACCENT_COLOR};">{fmt(hum_max)}</span> &nbsp; | &nbsp; Min: <span style="color:{ACCENT_COLOR};">{fmt(hum_min)}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background:{PANEL_BG}; padding:18px; border-radius:12px; margin-bottom:16px;">
            <h4 style="color:{ACCENT_COLOR};">Data Preview</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_visual.head(30), use_container_width=True)

# ---- PAGE 2: VISUALIZATIONS ----
elif page == "Visualizations":
    st.title("Weather Data Visualizations")
    st.markdown("Explore the main weather statistics interactively:")

    # 1. Temperature Over Time
    if 'Temp - °C' in df_visual.columns:
        st.subheader("Temperature Over Time")
        fig1 = px.line(
            df_visual,
            x=df_visual.index,
            y='Temp - °C',
            title="Temperature Over Time",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig1.update_layout(
            template=None,
            title_font=dict(size=24, color=ACCENT_COLOR, family='Arial Black'),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR, family='Segoe UI'),
            xaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Date', font=dict(size=18, color=ACCENT_COLOR))
            ),
            yaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Temperature (°C)', font=dict(size=18, color=ACCENT_COLOR))
            ),
            legend=dict(
                bgcolor=PANEL_BG,
                font=dict(size=14, color=ACCENT_COLOR),
                bordercolor='#26344d',
                borderwidth=2
            )
        )
        fig1.update_traces(line=dict(width=3))
        st.plotly_chart(fig1, use_container_width=True)
        st.info("Temperature trends show seasonal variations and highlight periods of extreme weather.")

    # 2. Humidity Distribution
    if 'Hum - %' in df_visual.columns:
        st.subheader("Humidity Distribution")
        fig2 = px.histogram(
            df_visual,
            x='Hum - %',
            nbins=50,
            title='Humidity Distribution',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig2.update_layout(
            template=None,
            title_font=dict(size=24, color=ACCENT_COLOR, family='Arial Black'),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR, family='Segoe UI'),
            xaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Humidity (%)', font=dict(size=18, color=ACCENT_COLOR))
            ),
            yaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Frequency', font=dict(size=18, color=ACCENT_COLOR))
            ),
            legend=dict(
                bgcolor=PANEL_BG,
                font=dict(size=14, color=ACCENT_COLOR),
                bordercolor='#26344d',
                borderwidth=2
            )
        )
        fig2.update_traces(marker_line_color=ACCENT_COLOR, marker_line_width=1)
        st.plotly_chart(fig2, use_container_width=True)
        st.info("Humidity levels are mostly distributed between 40% and 80%, indicating moderate to high moisture in the air throughout the dataset.")

    # 3. Wind Rose Diagram
    if 'Wind Speed - m/s' in df_visual.columns and 'Wind Direction' in df_visual.columns:
        st.subheader("Wind Rose Diagram")
        fig3 = px.scatter_polar(
            df_visual,
            r='Wind Speed - m/s',
            theta='Wind Direction',
            color='Wind Speed - m/s',
            color_continuous_scale=px.colors.sequential.Blues,
            title='Wind Rose Diagram'
        )
        fig3.update_layout(
            template=None,
            title=dict(
                text='Wind Rose Diagram',
                font=dict(size=24, color=ACCENT_COLOR, family='Arial Black')
            ),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR, family='Segoe UI'),
            polar=dict(
                bgcolor='#232a3a',
                angularaxis=dict(
                    gridcolor='#26344d',
                    tickfont=dict(color=ACCENT_COLOR, size=14)
                ),
                radialaxis=dict(
                    gridcolor='#26344d',
                    tickfont=dict(color=ACCENT_COLOR, size=14),
                    title=dict(text='Wind Speed (m/s)', font=dict(size=16, color=ACCENT_COLOR))
                )
            ),
            coloraxis_colorbar=dict(
                title=dict(text='Wind Speed (m/s)', font=dict(color=ACCENT_COLOR, size=16)),
                tickfont=dict(color=ACCENT_COLOR, size=12),
                bgcolor='#232a3a'
            )
        )
        fig3.update_traces(marker=dict(size=6, line=dict(width=1, color=FONT_COLOR)))
        st.plotly_chart(fig3, use_container_width=True)
        st.info("Wind rose shows dominant wind directions and speeds, helping identify prevailing weather patterns.")

    # 4. Rainfall Over Time
    if 'Rain - mm' in df_visual.columns:
        st.subheader("Rainfall Over Time")
        fig4 = px.line(
            df_visual,
            x=df_visual.index,
            y='Rain - mm',
            title="Rainfall Over Time",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig4.update_layout(
            template=None,
            title_font=dict(size=24, color=ACCENT_COLOR, family='Arial Black'),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR, family='Segoe UI'),
            xaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Date', font=dict(size=18, color=ACCENT_COLOR))
            ),
            yaxis=dict(
                gridcolor='#26344d',
                zerolinecolor='#26344d',
                title=dict(text='Rain (mm)', font=dict(size=18, color=ACCENT_COLOR))
            ),
            legend=dict(
                bgcolor=PANEL_BG,
                font=dict(size=14, color=ACCENT_COLOR),
                bordercolor='#26344d',
                borderwidth=2
            )
        )
        fig4.update_traces(line=dict(width=3))
        st.plotly_chart(fig4, use_container_width=True)
        st.info("Rainfall patterns indicate wet and dry periods, revealing seasonal changes and potential drought or flood events.")

    # 5. Yearly Mean Statistics
    if 'Year' not in df_visual.columns:
        df_visual['Year'] = df_visual.index.year
    yearly_stats = df_visual.groupby('Year').agg({
        'Temp - °C': ['mean', 'min', 'max'] if 'Temp - °C' in df_visual.columns else [],
        'Hum - %': ['mean', 'min', 'max'] if 'Hum - %' in df_visual.columns else [],
        'Wind Speed - m/s': ['mean', 'min', 'max'] if 'Wind Speed - m/s' in df_visual.columns else [],
        'Rain - mm': ['mean', 'min', 'max'] if 'Rain - mm' in df_visual.columns else [],
    })
    if not yearly_stats.empty:
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
        yearly_stats = yearly_stats.reset_index()
        metrics = [col for col in yearly_stats.columns if col != 'Year']
        colors = px.colors.sequential.Blues[:len(metrics)]
        fig5 = go.Figure()
        for metric, color in zip(metrics, colors):
            fig5.add_trace(go.Scatter(
                x=yearly_stats['Year'],
                y=yearly_stats[metric],
                mode='lines+markers',
                name=metric.replace('_mean', ''),
                line=dict(width=3, color=color)
            ))
        fig5.update_layout(
            title=dict(
                text='Yearly Mean Statistics',
                font=dict(size=24, color=ACCENT_COLOR, family='Arial Black')
            ),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=BG_COLOR,
            font=dict(color=FONT_COLOR, family='Segoe UI'),
            xaxis=dict(
                gridcolor='#26344d',
                title=dict(text='Year', font=dict(size=18, color=ACCENT_COLOR))
            ),
            yaxis=dict(
                gridcolor='#26344d',
                title=dict(text='Mean Value', font=dict(size=18, color=ACCENT_COLOR))
            ),
            legend=dict(
                bgcolor=PANEL_BG,
                font=dict(size=14, color=ACCENT_COLOR),
                bordercolor='#26344d',
                borderwidth=2
            )
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.info("Yearly averages reveal long-term trends and climate shifts.")

# ---- PAGE 3: MODEL DEPLOYMENT ----
elif page == "Model Deployment":
    st.title("Temperature Prediction Model")
    st.markdown(
        f"""
        <div style="background:{PANEL_BG}; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h2 style="color:{ACCENT_COLOR};">Forecast the Temperature</h2>
        <p>Enter current weather conditions below to predict the temperature using our trained model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input fields for model features
    col1, col2, col3, col4 = st.columns(4)
    barometer = col1.number_input("Barometer - mm Hg", value=760.0)
    humidity = col2.number_input("Hum - %", value=50.0)
    wind_speed = col3.number_input("Wind Speed - m/s", value=5.0)
    rain = col4.number_input("Rain - mm", value=0.0)
    heat_index = st.number_input("Heat Index - °C", value=30.0)
    season = st.selectbox("Season", options=[1, 2, 3, 4], format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x-1])
    day_night = st.selectbox("Day or Night", options=[0, 1], format_func=lambda x: "Day" if x == 0 else "Night")
    wind_direction = st.selectbox("Wind Direction", options=wind_dir_options)

    # Calculate Wind_Dir_sin and Wind_Dir_cos
    wind_dir_deg = direction_map[wind_direction]
    wind_dir_rad = np.deg2rad(wind_dir_deg)
    wind_dir_sin = np.sin(wind_dir_rad)
    wind_dir_cos = np.cos(wind_dir_rad)

    # Encode Wind_Direction with ordinal integer for model
    wind_direction_encoded = wind_dir_label_map[wind_direction]

    # Model expects features in this order:
    # ['Barometer - mm Hg', 'Hum - %', 'Wind Speed - m/s', 'Rain - mm',
    #  'Heat Index - °C', 'Season', 'Day_Night', 'Wind_Dir_sin', 'Wind_Dir_cos', 'Wind_Direction']
    input_features = np.array([[barometer, humidity, wind_speed, rain, heat_index, season, day_night, wind_dir_sin, wind_dir_cos, wind_direction_encoded]])

    predict_btn = st.button("Predict Temperature", type="primary")
    if predict_btn:
        try:
            prediction = model.predict(input_features)
            st.markdown(
                f"""
                <div style="background:linear-gradient(90deg, #1c2234 60%, #212a40 100%); padding:32px; border-radius:18px; border:2px solid {ACCENT_COLOR}; box-shadow:0px 4px 20px #12162044; margin-top:12px; text-align:center;">
                    <h2 style="color:{ACCENT_COLOR}; font-family:'Arial Black', 'Segoe UI'; font-size:40px;">🌡️ Predicted Temperature</h2>
                    <div style="font-size:56px; color:{ACCENT_COLOR}; font-weight:bold; letter-spacing:2px; margin-top:12px;">{prediction[0]:.2f} °C</div>
                    <div style="font-size:22px;color:{FONT_COLOR};margin-top:18px;">Based on your inputs, the model forecasts this temperature. Stay prepared!</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown(
                f"""
                <div class='stError'>
                    ❌ Prediction failed. Error: {e}
                </div>
                """,
                unsafe_allow_html=True
            )