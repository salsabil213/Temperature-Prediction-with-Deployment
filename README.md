# 🌡️ Temperature Prediction with Deployment

A comprehensive weather analysis and temperature prediction system built with Streamlit, featuring interactive visualizations and machine learning-powered forecasting using MX Nizanda weather station data (2006-2024).

## 🌟 Features

### 📊 Multi-Page Dashboard
- **Project Overview**: Key statistics and project insights
- **Data Visualizations**: Interactive charts with weather patterns
- **Temperature Prediction**: Real-time ML-powered temperature forecasting

### 🎨 Modern UI/UX
- Weather-themed dark interface with blue gradient styling
- Responsive design with intuitive navigation
- Interactive Plotly visualizations
- Real-time prediction interface

### 🤖 Machine Learning
- Random Forest temperature prediction model
- Trained on 33,000+ weather observations
- 7 meteorological features for accurate forecasting
- Model performance metrics and confidence indicators

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/salsabil213/Temperature-Prediction-with-Deployment.git
cd Temperature-Prediction-with-Deployment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

4. **Access the dashboard**
Open your browser and navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your forked repository
   - Set the main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Your app will be live** at `https://[your-app-name].streamlit.app`

### Alternative Deployment Options

#### Heroku
```bash
# Install Heroku CLI and login
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📁 Project Structure

```
Temperature-Prediction-with-Deployment/
├── streamlit_app.py                 # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── df_model_after_Pettit.csv      # Processed weather data (primary)
├── df_model_before_Pettit.csv     # Raw weather data
├── model.pkl                       # Trained ML model (auto-generated)
└── MX Nizanda weather station...   # Original Excel data
```

## 📊 Data Features

The weather dataset includes the following meteorological parameters:

| Feature | Description | Unit |
|---------|-------------|------|
| Barometer | Atmospheric pressure | mm Hg |
| Humidity | Relative humidity | % |
| Wind Speed | Wind velocity | m/s |
| Rain | Precipitation amount | mm |
| Heat Index | Perceived temperature | °C |
| Temperature | Air temperature (target) | °C |
| Season | Seasonal indicator | 0/1 |
| Day_Night | Time period | 0/1 |
| Wind Direction | Wind compass direction | degrees |

## 🎯 Model Performance

- **Algorithm**: Random Forest Regressor
- **Training Data**: 33,000+ hourly observations
- **Features**: 7 meteorological parameters
- **Accuracy**: R² score and MSE calculated dynamically
- **Real-time Predictions**: Interactive parameter adjustment

## 🎨 Design Theme

The application features a modern weather-themed design with:

- **Color Palette**: Blue gradients (#a6c7ff, #8cb8ff, #b2c7e0)
- **Dark Theme**: Professional appearance with high contrast
- **Weather Icons**: Intuitive visual elements
- **Responsive Layout**: Works on desktop and mobile devices

## 🔧 Technical Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas, NumPy for data handling
- Plotly for interactive visualizations
- Scikit-learn for machine learning
- 512MB+ RAM recommended

## 📈 Usage Examples

### Temperature Prediction
1. Navigate to the "Temperature Prediction" page
2. Adjust weather parameters using interactive sliders
3. Click "Predict Temperature" for instant results
4. View prediction confidence and temperature category

### Data Exploration
1. Visit "Data Visualizations" page
2. Explore interactive charts and patterns
3. Read insights below each visualization
4. Understand weather trends over time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MX Nizanda Weather Station for providing comprehensive meteorological data
- Streamlit team for the excellent framework
- Plotly for powerful visualization capabilities
- Scikit-learn for robust machine learning tools

## 📞 Support

For questions or support, please:
- Open an issue on GitHub
- Check the [Streamlit documentation](https://docs.streamlit.io)
- Review the application logs for troubleshooting

---

**Live Demo**: [Your Streamlit App URL]

**Last Updated**: December 2024