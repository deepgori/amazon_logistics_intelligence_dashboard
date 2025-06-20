# Google Routes API Setup Guide

## Overview
This dashboard now includes Google Routes API integration to provide real-time routing analysis and delay prediction based on actual road conditions, traffic, and optimal routes between fulfillment centers and delivery destinations.

## Features Added

### 1. **Real FC Distance Calculation**
- Calculates actual haversine distances between fulfillment centers and delivery destinations
- Identifies the nearest FC for each order
- Provides distance-based delay risk assessment

### 2. **Google Routes API Integration**
- Real-time route optimization using Google's traffic-aware routing
- Actual road distance vs. direct distance analysis
- Traffic condition assessment and speed analysis
- Route efficiency scoring

### 3. **Enhanced Delay Prediction**
- Integrates routing data into ML prediction model
- Considers traffic incidents and route efficiency
- Provides routing-specific delay recommendations

### 4. **AI Assistant Routing Analysis**
- New question: "🗺️ How can we optimize routing and FC distances?"
- FC load distribution analysis
- Route optimization recommendations
- Cost-benefit analysis for routing improvements

## Setup Instructions

### 1. Get Google Routes API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable the **Routes API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Routes API"
   - Click "Enable"
4. Create API credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the API key

### 2. Set Environment Variable
Set your Google Routes API key as an environment variable:

```bash
# On macOS/Linux
export GOOGLE_ROUTES_API_KEY="AIzaSyCdyYtpYe_5PGZgEutZWmfosL44iqtQQ5w"

```

### 3. Verify Setup
1. Start the dashboard: `streamlit run scripts/web_app.py`
2. Go to "ML Prediction Demo" page
3. Enter order details and check the "Routing Analysis" section
4. Go to "AI Dispatcher Assistant" and select "🗺️ How can we optimize routing and FC distances?"

## What the Integration Analyzes

### **FC Distance Analysis**
- Calculates nearest FC for each destination
- Identifies long-distance orders (>500km)
- Analyzes FC load distribution

### **Route Optimization**
- Compares direct distance vs. actual road distance
- Identifies inefficient routes (efficiency ratio > 1.3)
- Analyzes traffic conditions and speed

### **Delay Factors**
- Long travel times (>8 hours)
- Slow average speeds (<40 km/h)
- Traffic congestion detection
- Route inefficiency scoring

### **Recommendations**
- FC placement optimization
- Route optimization strategies
- Load balancing across FCs
- Expedited shipping for long distances

## Cost Considerations

### **Google Routes API Pricing**
- $5 per 1,000 requests
- Traffic-aware routing: $5 per 1,000 requests
- Real-time traffic data included

### **Estimated Usage**
- Sample analysis: ~5 routes per analysis
- Daily usage: ~100-500 requests
- Monthly cost: $5-25 (depending on usage)

## Benefits

### **Improved Accuracy**
- Real-world routing instead of straight-line distances
- Traffic-aware delay predictions
- Actual road conditions consideration

### **Better Optimization**
- Identify inefficient routes
- Optimize FC placement
- Reduce delivery delays

### **Cost Savings**
- Route optimization can reduce delivery costs
- Better FC placement reduces average distances
- Improved efficiency reduces operational costs

## Troubleshooting

### **API Key Issues**
- Ensure API key is set correctly
- Check that Routes API is enabled
- Verify billing is set up in Google Cloud Console

### **No Routing Data**
- If Google Routes API is not available, the system falls back to haversine distance calculation
- All features still work with fallback data
- Routing analysis will show "API not configured" message

### **Performance Issues**
- Google Routes API has rate limits
- Consider caching results for repeated routes
- Implement request throttling if needed

## Future Enhancements

### **Planned Features**
- Route caching to reduce API calls
- Batch route optimization
- Historical route performance tracking
- Multi-stop route optimization
- Alternative route suggestions

### **Advanced Analytics**
- Route performance over time
- Seasonal routing patterns
- Weather impact on routes
- Predictive route optimization 