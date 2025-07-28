# 🏦 Churn Guard AI - Customer Retention Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

**Churn Guard AI** is an intelligent customer retention prediction system that uses Artificial Neural Networks to predict whether bank customers are likely to churn (leave the bank). The system provides an intuitive web interface for real-time predictions and actionable business insights.

### 🎯 Key Features

- **AI-Powered Predictions**: Neural network with 85.05% accuracy
- **Interactive Web Interface**: Beautiful Streamlit dashboard
- **Real-time Analysis**: Instant churn probability calculations
- **Visual Risk Assessment**: Color-coded gauge charts and metrics
- **Business Insights**: Actionable recommendations for customer retention
- **Professional UI**: Modern design with intuitive user experience

## 🚀 Demo

![Churn Guard AI Demo](./images/image.png)

*Screenshot of the Churn Guard AI web application showing the interactive dashboard with customer input form, prediction results, and risk visualization.*

## 📊 Model Performance

- **Accuracy**: 85.05% on test dataset
- **Model Type**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Features**: 11 customer attributes
- **Dataset**: 10,000 bank customer records

## 🛠️ Technology Stack

- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, matplotlib
- **Environment**: Python 3.11+

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ayeshowcode/CustomerRetentionAI.git
cd CustomerRetentionAI
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow streamlit pandas numpy scikit-learn plotly matplotlib seaborn
```

### 4. Train the Model (First Time Setup)

Open and run the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

**Important**: Run all cells in the notebook to:
- Load and preprocess the data
- Train the neural network model
- Save the trained model (`churn_model.h5`) and scaler (`scaler.pkl`)

### 5. Launch the Web Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
CustomerRetentionAI/
├── main.ipynb              # Jupyter notebook with ML pipeline
├── streamlit_app.py        # Streamlit web application
├── Churn_Modelling.csv     # Dataset (10,000 customer records)
├── churn_model.h5          # Trained neural network model
├── scaler.pkl              # Data preprocessing scaler
├── README.md               # Project documentation
└── .venv/                  # Virtual environment (created after setup)
```

## 🧠 How It Works

### 1. Data Processing
- **Input Features**: 11 customer attributes (demographics, financial, banking)
- **Preprocessing**: StandardScaler normalization
- **Encoding**: One-hot encoding for categorical variables

### 2. Neural Network Architecture
```python
Model: Sequential
├── Input Layer (11 features)
├── Hidden Layer (3 neurons, sigmoid activation)
├── Output Layer (1 neuron, sigmoid activation)
└── Optimizer: Adam, Loss: binary_crossentropy
```

### 3. Prediction Pipeline
1. User inputs customer data through web interface
2. Data is preprocessed using saved scaler
3. Neural network generates churn probability
4. Results displayed with visual risk assessment

## 🎨 Web Application Features

### 📊 Dashboard Components

1. **Customer Input Form**: Sidebar with organized input fields
2. **Prediction Metrics**: Three-column display showing:
   - 🎯 Churn Probability (percentage)
   - 🤖 AI Prediction (Will Churn/Will Stay)
   - 🔍 Confidence Level
3. **Risk Visualization**: Interactive gauge chart with color zones:
   - 🟢 Green (0-30%): Low Risk
   - 🟡 Yellow (30-70%): Medium Risk
   - 🔴 Red (70-100%): High Risk
4. **Business Recommendations**: Actionable insights based on risk level
5. **Customer Profile Summary**: Detailed breakdown of input data

### 🎯 Input Features

| Category | Features |
|----------|----------|
| **Demographics** | Age, Gender |
| **Location** | Country (France, Germany, Spain) |
| **Financial** | Credit Score, Account Balance, Estimated Salary |
| **Banking** | Years with Bank, Number of Products, Credit Card, Active Member |

## 📈 Usage Examples

### Example 1: Low Risk Customer
```
Age: 35, Credit Score: 750, Balance: $50,000
Years with Bank: 5, Active Member: Yes
→ Prediction: 15% churn probability (Low Risk)
→ Recommendation: Standard service level
```

### Example 2: High Risk Customer
```
Age: 55, Credit Score: 400, Balance: $0
Years with Bank: 1, Active Member: No
→ Prediction: 85% churn probability (High Risk)
→ Recommendation: Immediate retention action needed
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ayesh** - *Initial work* - [@ayeshowcode](https://github.com/ayeshowcode)

---

*Built with ❤️ for better customer retention*
