# Renewable Energy & Electric Load Forecasting Dashboard ⚡️
This project is a comprehensive forecasting tool designed to predict renewable energy generation and electric load in Maharashtra, India. It leverages a machine learning model and provides insights through an interactive web dashboard, demonstrating a full-stack data science application.

## 🌟 Features

* **ML-Powered Forecasting**: Utilizes a fine-tuned **XGBoost Regressor** model to predict daily electric load based on weather patterns and categorical data.
* **Interactive Frontend**: A user-friendly dashboard built with **Streamlit** allows for real-time forecast generation by inputting custom parameters.
* **REST API Backend**: A robust **Flask** API serves the machine learning model, making it scalable and easy to connect with various clients.
* **Energy Balance Analysis**: Provides crucial insights by visualizing the historical **surplus or shortage** of renewable energy against the electric load.
* **Comparative Analytics**: Features dynamic charts comparing energy load and weather patterns across different cities and seasons in Maharashtra.

## 🛠️ Tech Stack

* **Backend**: Python, Flask, XGBoost, Pandas, Scikit-learn
* **Frontend**: Streamlit, Plotly, Matplotlib, Requests
* **Data Science**: Jupyter Notebook, Feature Engineering, Hyperparameter Tuning

## 🏗️ System Architecture

The application operates on a simple and effective client-server model:

1.  **Frontend (Streamlit)**: The user interacts with the web dashboard, enters input data, and clicks "Get Prediction". The frontend sends this data as a JSON object to the backend.
2.  **Backend (Flask API)**: The API receives the JSON data, preprocesses it to match the model's expected input, and feeds it into the trained XGBoost model.
3.  **Prediction**: The model returns a prediction, which the Flask API then sends back to the frontend as a JSON response.
4.  **Visualization**: The Streamlit frontend receives the prediction and displays it to the user, along with other historical data visualizations.

## 🚀 Setup and Installation

Follow these steps to run the project locally.

### 1. Prerequisites

* Python 3.9+
* Git

### 2. Create `requirements.txt`

Before you can run the installation, you need to generate a `requirements.txt` file from your project's virtual environment (`.venv`). This file lists all the necessary libraries.

**In your terminal, activate your virtual environment and run:**
```bash
pip freeze > requirements.txt
```
This will create the `requirements.txt` file. Make sure to add and commit this file to your Git repository.

### 3. Clone & Install

```bash
# Clone the repository
git clone [https://github.com/your-username/Your-Repo-Name.git](https://github.com/your-username/Your-Repo-Name.git)
cd Your-Repo-Name

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt
```

### 4. Run the Application

You need to run the backend and frontend simultaneously in two separate terminals.

**Terminal 1: Start the Backend**
```bash
python backend.py
```
*The backend will be running at `http://127.0.0.1:5000`*

**Terminal 2: Start the Frontend**
```bash
streamlit run frontend.py
```
*The application will open automatically in your web browser.*

## 🔮 Future Work

As outlined in our project thesis, future development will focus on:
* **Advanced Deep Learning Models**: Building and integrating an **LSTM (Long Short-Term Memory)** network using the available hourly dataset to capture more complex time-series patterns.
* **Real-Time Data Integration**: Incorporating live weather feeds to provide truly real-time forecasts.
* **Hybrid Modeling**: Combining our machine learning model with physical simulation models for enhanced accuracy.

## 👥 Project Team

This project was developed by:
* Rajeev Nair
* Rahul Singh
* Nair Sivabalkrishnan

Under the guidance of Prof. Ranjana Singh.