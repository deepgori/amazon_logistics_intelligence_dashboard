# Amazon Logistics Intelligence Dashboard

A comprehensive logistics analytics platform built with Streamlit that provides real-time insights into Amazon delivery operations, cost analysis, and predictive analytics for delivery delays.

## 🚀 Features

### 📊 Dashboard Sections
- **Prime Performance**: Comprehensive delivery metrics and carrier analysis
- **Last-Mile Operations**: Real-time route optimization and vehicle efficiency  
- **Cost Efficiency Analysis**: Cost analysis and optimization insights
- **Amazon Purchase Trends**: Product sales and category analysis
- **ML Prediction Demo**: Real-time delivery delay prediction
- **AI Dispatcher Assistant**: Intelligent logistics advisor with guided questions

### 🤖 AI & ML Capabilities
- **Delivery Delay Prediction**: Machine learning model for predicting delivery delays
- **Natural Language Queries**: Ask questions about logistics data in plain English
- **Proactive Alerts**: Real-time monitoring of internal and external factors
- **Route Optimization**: Google Routes API integration for traffic-aware routing

### 📈 Data Integration
- **Real-time APIs**: Weather alerts, news alerts, traffic data
- **Historical Data**: Amazon purchase history and simulated logistics data
- **External Sources**: OpenWeatherMap, NewsAPI, Google Maps APIs

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Altair, Plotly
- **Machine Learning**: Scikit-learn, SHAP
- **APIs**: Google Routes, OpenWeatherMap, NewsAPI
- **Database**: PostgreSQL (optional)

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- Google Routes API key (optional, for route optimization)
- OpenWeatherMap API key (optional, for weather data)
- NewsAPI key (optional, for news alerts)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AWS_Project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   # API Keys (optional - app works without them)
   GOOGLE_ROUTES_API_KEY=your_google_routes_api_key
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
   NEWSAPI_API_KEY=your_newsapi_key
   MAPS_API_KEY=your_google_maps_api_key
   
   # Database (optional)
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=amazon_delivery_db
   DB_USER=postgres
   DB_PASSWORD=your_database_password
   ```

5. **Set up Streamlit secrets (alternative to .env)**
   Create `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_ROUTES_API_KEY = "your_google_routes_api_key"
   OPENWEATHERMAP_API_KEY = "your_openweathermap_api_key"
   NEWSAPI_API_KEY = "your_newsapi_key"
   MAPS_API_KEY = "your_google_maps_api_key"
   ```

## 🚀 Usage

### Start the Dashboard
```bash
streamlit run scripts/web_app.py
```

### Start the ML Prediction API (optional)
```bash
cd scripts
uvicorn ml_prediction_api:app --reload --port 8000
```

### Run Data Pipeline
```bash
python scripts/run_pipeline.py
```

## 📁 Project Structure

```
AWS_Project/
├── scripts/
│   ├── web_app.py                 # Main Streamlit application
│   ├── pages/                     # Dashboard page modules
│   │   ├── prime_performance.py
│   │   ├── last_mile_operations.py
│   │   └── cost_efficiency_analysis.py
│   ├── utils/                     # Utility functions
│   ├── config/                    # Configuration files
│   ├── ml_prediction_api.py       # FastAPI for ML predictions
│   ├── data_generator.py          # Simulated data generation
│   └── api_integrator.py          # External API integrations
├── data/                          # Data files (not in git)
├── models/                        # ML models (not in git)
├── .streamlit/                    # Streamlit configuration
├── config/                        # Project configuration
└── requirements.txt               # Python dependencies
```

## 🔒 Security Considerations

### API Keys
- **Never commit API keys to version control**
- Use environment variables or Streamlit secrets
- Rotate keys regularly
- Use least-privilege access

### Data Privacy
- All data is simulated or anonymized
- No real customer data is used
- External APIs are optional

### Database Security
- Use strong passwords
- Enable SSL connections
- Restrict network access
- Regular security updates

## 🧪 Testing

### Run Basic Tests
```bash
python -c "import sys; sys.path.append('.'); from scripts.pages.cost_efficiency_analysis import render_page; print('✅ Cost Analysis page works')"
```

### Test API Endpoints
```bash
curl http://localhost:8000/health
```

## 📊 Data Sources

### Simulated Data
- **Orders**: 15,000+ simulated Amazon orders
- **Carriers**: AMZL, UPS, FedEx, USPS
- **Cities**: 15 major US cities
- **Time Period**: 6 months of historical data

### External APIs
- **Weather**: OpenWeatherMap API
- **News**: NewsAPI
- **Traffic**: Google Maps Directions API
- **Routing**: Google Routes API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 🆘 Troubleshooting

### Common Issues

**API Key Errors**
- Ensure API keys are set in environment variables or Streamlit secrets
- Check API key permissions and quotas
- Verify API key format

**Database Connection Issues**
- Check database credentials
- Ensure PostgreSQL is running
- Verify network connectivity

**Missing Dependencies**
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

### Getting Help
- Check the logs in `app.log` and `data_processing.log`
- Review the `IMPROVEMENTS_GUIDE.md` for known issues
- Open an issue with detailed error information

## 🔄 Updates

### Recent Changes
- ✅ Modular code structure with separate page modules
- ✅ Enhanced security with proper API key management
- ✅ Simplified cost efficiency analysis
- ✅ Improved error handling and logging
- ✅ Comprehensive documentation

### Roadmap
- 🔄 Advanced filtering and search
- 🔄 Real-time data streaming
- 🔄 Custom dashboard builder
- 🔄 Mobile-responsive design
- 🔄 Export functionality

---

**Note**: This is a demonstration project. For production use, implement proper security measures, error handling, and monitoring. 