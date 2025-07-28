import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os

st.set_page_config(
    page_title="🏦 Churn Guard AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Churn Guard AI")
st.markdown("""
### 🤖 AI-Powered Customer Retention Tool
This app uses an **Artificial Neural Network** to predict whether a bank customer will churn (leave the bank).
Enter customer details below to get an instant prediction!
""")

st.sidebar.header("📝 Customer Information")
st.sidebar.markdown("Fill in the customer details:")

def get_user_input():
    st.sidebar.subheader("👤 Demographics")
    age = st.sidebar.slider("Age", 18, 95, 40)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    
    st.sidebar.subheader("🌍 Location")
    geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
    
    st.sidebar.subheader("💰 Financial Details")
    credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 10000.0, 200000.0, 50000.0, step=1000.0)
    
    st.sidebar.subheader("🏦 Banking Details")
    tenure = st.sidebar.slider("Years with Bank", 0, 10, 3)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"])
    is_active_member = st.sidebar.selectbox("Active Member?", ["No", "Yes"])
    
    features = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0
    }
    
    return pd.DataFrame([features])

@st.cache_resource #so model load only once
def load_model_and_scaler():
    try:
        # Try to load saved model and scaler
        model = tf.keras.models.load_model('churn_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except:
        st.warning("⚠️ Saved model not found. Please train the model first by running your notebook!")
        return None, None, False

def make_prediction(model, scaler, user_data):
    # Scale the features
    user_data_scaled = scaler.transform(user_data)
    
    prediction_prob = model.predict(user_data_scaled)[0][0] #neural networks always return 2D array
    prediction_binary = 1 if prediction_prob > 0.5 else 0
    
    return prediction_prob, prediction_binary

def display_results(prob, binary_pred, threshold=0.5):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Churn Probability", 
            value=f"{prob:.1%}",
            delta=f"{'High Risk' if prob > threshold else 'Low Risk'}"
        )
    
    with col2:
        prediction_text = "Will Churn 😔" if binary_pred == 1 else "Will Stay 😊"
        st.metric(
            label="🤖 AI Prediction", 
            value=prediction_text
        )
    
    with col3:
        confidence = max(prob, 1-prob)
        st.metric(
            label="🔍 Confidence", 
            value=f"{confidence:.1%}"
        )

def create_gauge_chart(probability):
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Get user input
    user_data = get_user_input()
    
    # Load model
    model, scaler, model_loaded = load_model_and_scaler()
    
    if model_loaded:
        # Predict button
        if st.sidebar.button("🔮 Predict Churn", type="primary"):
            with st.spinner("🤖 AI is analyzing customer data..."):
                prob, binary_pred = make_prediction(model, scaler, user_data)
                
                # Display results
                st.subheader("🎯 Prediction Results")
                display_results(prob, binary_pred)
                
                # Visualization
                st.subheader("📊 Risk Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_gauge_chart(prob)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk interpretation
                    if prob < 0.3:
                        st.success("✅ **Low Churn Risk**\nCustomer is likely to stay with the bank.")
                        st.info("💡 **Recommendation**: Standard service level.")
                    elif prob < 0.7:
                        st.warning("⚠️ **Medium Churn Risk**\nCustomer might leave the bank.")
                        st.info("💡 **Recommendation**: Consider retention offers.")
                    else:
                        st.error("🚨 **High Churn Risk**\nCustomer is very likely to churn!")
                        st.info("💡 **Recommendation**: Immediate retention action needed!")
                
                # Detailed Analysis
                st.subheader("📋 Customer Profile Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Customer Details:**")
                    st.write(f"- Age: {user_data['Age'].iloc[0]} years")
                    st.write(f"- Credit Score: {user_data['CreditScore'].iloc[0]}")
                    st.write(f"- Account Balance: ${user_data['Balance'].iloc[0]:,.2f}")
                    st.write(f"- Estimated Salary: ${user_data['EstimatedSalary'].iloc[0]:,.2f}")
                
                with col2:
                    st.write("**Banking Relationship:**")
                    st.write(f"- Years with Bank: {user_data['Tenure'].iloc[0]}")
                    st.write(f"- Number of Products: {user_data['NumOfProducts'].iloc[0]}")
                    st.write(f"- Has Credit Card: {'Yes' if user_data['HasCrCard'].iloc[0] else 'No'}")
                    st.write(f"- Active Member: {'Yes' if user_data['IsActiveMember'].iloc[0] else 'No'}")
    
    else:
        st.error("❌ **Model not available**")
        st.info("""
        🔧 **Setup Instructions:**
        1. Run your Jupyter notebook to train the model
        2. Save the model and scaler (add this to your notebook):
        
        ```python
        # Save model and scaler for Streamlit app
        model.save('churn_model.h5')
        
        import pickle
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        ```
        """)

# 📖 About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### 🧠 How It Works
    This app uses an **Artificial Neural Network (ANN)** trained on bank customer data to predict churn probability.
    
    ### 🔍 Features Used
    - **Demographics**: Age, Gender, Geography
    - **Financial**: Credit Score, Balance, Salary
    - **Banking**: Tenure, Products, Credit Card, Activity
    
    ### 🎯 Model Performance
    - **Accuracy**: 85.05% on test data
    - **Type**: Binary Classification
    - **Framework**: TensorFlow/Keras
    
    ### 💡 Business Value
    Banks can use this tool to:
    - Identify at-risk customers
    - Plan retention campaigns
    - Reduce customer churn
    - Increase profitability
    """)

# Run the app
if __name__ == "__main__":
    main()

# 🎨 Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #FF5252;
        color: white;
    }
    
    .metric-container {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)
