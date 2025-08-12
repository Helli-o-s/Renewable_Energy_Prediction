import streamlit as st
import requests
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Maharashtra Energy Forecaster",
    page_icon="⚡️",
    layout="wide"
)

# --- Backend API URL ---
API_URL = "http://127.0.0.1:5000/predict"

# --- Data and Model Loading ---
@st.cache_data
def load_data():
    """Loads and preprocesses the historical dataset."""
    df = pd.read_csv(
        "maharashtra_2024_daily_energy_forecasting_dataset.csv",
        parse_dates=['Date'],
        encoding='latin1'
    )
    # FIX: Rename the problematic column name
    df.rename(columns={'Temperature (Â°C)': 'Temperature (C)'}, inplace=True)
    
    df.set_index('Date', inplace=True)
    # Calculate Total Generation and Energy Balance
    df['Total Generation (MW)'] = df['Wind Power (MW)'] + df['Solar Power (MW)'] + df['Hydro Power (MW)']
    df['Energy Balance (MW)'] = df['Total Generation (MW)'] - df['Electric Load (MW)']
    return df

@st.cache_resource
def load_model():
    """Loads the XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model("best_xgb_model.json")
    return model

df_historical = load_data()
model = load_model()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("🔮 Prediction Inputs")
    with st.form("prediction_form"):
        city = st.selectbox("Select City", ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"])
        load_type = st.selectbox("Select Load Type", ["Residential", "Industrial", "Mixed"])
        st.markdown("---")
        temperature = st.slider("Temperature (°C)", 10.0, 50.0, 30.0)
        humidity = st.slider("Humidity (%)", 20.0, 100.0, 60.0)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 5.5)
        solar_radiation = st.slider("Solar Radiation (W/m²)", 1000.0, 6000.0, 3000.0)
        submitted = st.form_submit_button("Get Prediction")


# --- Main App Layout ---
st.title('Maharashtra Energy Forecaster ⚡️')

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Energy Balance", "🏙️ City Comparison"])

# --- Tab 1: Forecast ---
with tab1:
    st.header("Forecast Result")
    result_placeholder = st.empty()

    if submitted:
        city_feature = "City_" + city
        load_type_feature = "Load Type_" + load_type
        input_data = {
            "Temperature (C)": temperature,
            "Humidity (%)": humidity,
            "Wind Speed (m/s)": wind_speed,
            "Solar Radiation (W/m²)": solar_radiation,
            city_feature: 1,
            load_type_feature: 1
        }
        with st.spinner("Forecasting..."):
            try:
                response = requests.post(API_URL, json=input_data)
                response.raise_for_status()
                result = response.json()
                predicted_load = result['prediction']
                result_placeholder.metric(label="Predicted Electric Load", value=f"{predicted_load:.2f} MW")
            except requests.exceptions.RequestException as e:
                result_placeholder.error(f"Could not connect to the backend: {e}")
    else:
        result_placeholder.info("Enter details in the sidebar and click 'Get Prediction' to see the forecast.")

    st.markdown("---")
    st.header("Historical Electric Load vs. Generation (2024)")
    st.line_chart(df_historical[['Electric Load (MW)', 'Total Generation (MW)']])

# --- Tab 2: Energy Balance ---
with tab2:
    st.header("Historical Surplus vs. Shortage")
    st.write("This chart shows the daily difference between total renewable energy generation and electric load. Green indicates a surplus, and red indicates a shortage.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_historical.index, y=df_historical['Energy Balance (MW)'],
                             mode='lines', name='Energy Balance'))
    fig.add_hrect(y0=0, y1=df_historical['Energy Balance (MW)'].max(), line_width=0, fillcolor="green", opacity=0.2)
    fig.add_hrect(y0=df_historical['Energy Balance (MW)'].min(), y1=0, line_width=0, fillcolor="red", opacity=0.2)
    
    fig.update_layout(title="Energy Balance (Surplus/Shortage)", yaxis_title="Energy (MW)")
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: City Comparison ---
with tab3:
    st.header("Comparing Energy Load Across Cities")
    city_comparison = df_historical.pivot_table(index='Season', columns='City', values='Electric Load (MW)', aggfunc='mean')
    st.bar_chart(city_comparison)

    st.markdown("---")
    st.header("Explore Data by City")
    selected_city = st.selectbox("Select a City to View Detailed Data", df_historical['City'].unique())
    st.line_chart(df_historical[df_historical['City'] == selected_city][['Electric Load (MW)', 'Temperature (C)', 'Wind Power (MW)']])