import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# --- 1. Load the Saved Model and Data ---
# Load the trained XGBoost model
model = xgb.XGBRegressor()
model.load_model("best_xgb_model.json")

# Load the training columns
training_columns = joblib.load('training_columns.pkl')

# Load the test data (we'll use this to show the forecast)
# Make sure your original dataframe 'df' is available or load it from a csv
# For this example, let's assume 'X_test' and 'y_test' are available
# In a real app, you'd load and preprocess the data here.
# For simplicity, we'll just load the final test sets.
# You might need to save X_test and y_test to csv files and load them here.
# Example:
# X_test.to_csv('X_test.csv')
# y_test.to_csv('y_test.csv')
X_test = pd.read_csv('X_test.csv', index_col='Date', parse_dates=True)
y_test = pd.read_csv('y_test.csv', index_col='Date', parse_dates=True)

# Make predictions with the loaded model
y_pred = model.predict(X_test)


# --- 2. Build the Streamlit App ---
st.set_page_config(layout="wide")

# Title of the app
st.title('⚡️ Renewable Energy & Electric Load Forecasting Dashboard')

# --- 3. Display the Forecast ---
st.header('Electric Load Forecast vs. Actual')

# Create a dataframe for plotting
plot_df = pd.DataFrame({'Actual Load': y_test['Electric Load (MW)'], 'Predicted Load': y_pred}, index=y_test.index)

# Display a line chart
st.line_chart(plot_df)


# --- 4. Display Feature Importance ---
st.header('Feature Importance')
st.write("This chart shows which factors have the biggest impact on the electric load forecast.")

# Get feature importance from the model
feature_importance = pd.DataFrame({'feature': training_columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
st.pyplot(fig)