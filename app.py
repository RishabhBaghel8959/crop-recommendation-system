import streamlit as st
from crop_recommendation_model import CropRecommendationModel
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌾",
    layout="centered"
)

# No custom CSS - using Streamlit's default dark theme
# The green comes from Streamlit's default success messages and the dark background

# Header
st.title("🌾 AI-Based Crop Recommendation System")
st.markdown("### Get personalized crop recommendations based on your soil and climate conditions")

# Initialize model
@st.cache_resource
def load_model():
    model = CropRecommendationModel()
    df = model.load_and_preprocess_data(r"D:\Ankita Project\Crop_recommendation.csv")
    model.train_models(df)
    return model

with st.spinner('Loading AI model...'):
    model = load_model()

st.success("✅ Model loaded successfully! Ready to predict.")

st.markdown("---")

# Input Section
st.markdown("## 📝 Enter Soil & Climate Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients")
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=43.0, step=1.0)

with col2:
    st.subheader("🌤️ Climate Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.87, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.0, step=1.0)

# pH in separate row
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# Predict button
if st.button("🔍 Get Recommendation"):
    with st.spinner('Analyzing your inputs...'):
        result = model.predict_crop_and_variety(n, p, k, temperature, humidity, ph, rainfall)
    
    # Display results
    st.markdown("---")
    st.header("📋 Recommendation Results")
    
    # Metrics
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.metric("🌾 Recommended Crop", result['crop'].upper())
    
    with res_col2:
        st.metric("✅ Crop Confidence", f"{result['crop_confidence']:.1f}%")
    
    with res_col3:
        st.metric("🌱 Recommended Variety", result['variety'].upper())
    
    with res_col4:
        st.metric("✅ Variety Confidence", f"{result['variety_confidence']:.1f}%")
    
    # Explanation section
    if 'explanation' in result:
        exp = result['explanation']
        
        st.markdown("---")
        st.subheader("📖 Detailed Explanation")
        
        if 'recommendation_summary' in exp:
            st.info(exp['recommendation_summary'])
        
        # Two column layout for analysis
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            if 'why_this_crop' in exp:
                st.subheader("💡 Why This Crop?")
                why_crop = exp['why_this_crop']
                if isinstance(why_crop, list):
                    for item in why_crop:
                        st.write(f"✓ {item}")
                else:
                    st.write(why_crop)
        
        with analysis_col2:
            if 'cultivation_tips' in exp:
                st.subheader("🌱 Cultivation Tips")
                for tip in exp['cultivation_tips']:
                    st.write(f"• {tip}")
        
        # Condition Analysis
        if 'condition_analysis' in exp:
            st.markdown("---")
            st.subheader("🔍 Condition Analysis")
            
            condition_data = exp['condition_analysis']
            
            cond_col1, cond_col2 = st.columns(2)
            
            with cond_col1:
                st.success("**✅ Favorable Conditions:**")
                favorable_found = False
                
                if isinstance(condition_data, dict) and 'favorable' in condition_data:
                    favorable = condition_data['favorable']
                    if isinstance(favorable, list) and len(favorable) > 0:
                        for condition in favorable:
                            st.write(f"• {condition}")
                            favorable_found = True
                    elif isinstance(favorable, str) and favorable:
                        st.write(f"• {favorable}")
                        favorable_found = True
                
                if not favorable_found:
                    st.write("No specific favorable conditions identified.")
            
            with cond_col2:
                st.warning("**⚠️ Areas for Improvement:**")
                improvement_found = False
                
                if isinstance(condition_data, dict) and 'needs_attention' in condition_data:
                    needs_attention = condition_data['needs_attention']
                    if isinstance(needs_attention, list) and len(needs_attention) > 0:
                        for condition in needs_attention:
                            st.write(f"• {condition}")
                            improvement_found = True
                    elif isinstance(needs_attention, str) and needs_attention:
                        st.write(f"• {needs_attention}")
                        improvement_found = True
                
                if not improvement_found:
                    st.write("All conditions are favorable!")

# Footer
st.markdown("---")
st.markdown("**Developed by:** Ankita Singh Baghel, Aparna Singh Dubey, Ashish Singh Tomar, Amay Tiwari")
st.markdown("**Powered by:** Machine Learning & AI")
