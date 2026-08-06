import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Ultra-Modern CSS Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .header-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        text-align: center;
    }

    .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #e2e8f0 !important;
        border: 1px solid #6366f1 !important;
        border-radius: 8px !important;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .warning {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header UI
st.markdown("""
    <div class="header-card">
        <h1 style="color: #c084fc; margin-bottom: 0;">🩺 AI Breast Cancer Diagnostic Assistant</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 8px;">FastAPI Machine Learning Inference Pipeline</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.title("System Status")
    st.success("🟢 Model Loaded: Random Forest")
    st.info("⚡ API Server: `http://127.0.0.1:8000`")
    st.write("---")
    st.caption("Adjust individual feature values below for analysis.")

# Feature Names & Default Sample Values (From dataset)
features_info = [
    ("Mean Radius", 17.99), ("Mean Texture", 10.38), ("Mean Perimeter", 122.8), ("Mean Area", 1001.0),
    ("Mean Smoothness", 0.1184), ("Mean Compactness", 0.2776), ("Mean Concavity", 0.3001), ("Mean Concave Points", 0.1471),
    ("Mean Symmetry", 0.2419), ("Mean Fractal Dimension", 0.07871),
    ("Radius Error", 1.095), ("Texture Error", 0.9053), ("Perimeter Error", 8.589), ("Area Error", 153.4),
    ("Smoothness Error", 0.006399), ("Compactness Error", 0.04904), ("Concavity Error", 0.05373), ("Concave Points Error", 0.01587),
    ("Symmetry Error", 0.03003), ("Fractal Dimension Error", 0.006193),
    ("Worst Radius", 25.38), ("Worst Texture", 17.33), ("Worst Perimeter", 184.6), ("Worst Area", 2019.0),
    ("Worst Smoothness", 0.1622), ("Worst Compactness", 0.6656), ("Worst Concavity", 0.7119), ("Worst Concave Points", 0.2654),
    ("Worst Symmetry", 0.4601), ("Worst Fractal Dimension", 0.1189)
]

st.subheader("📋 Enter Clinical Measurements")

# Organized Into 3 Tabs for Clean Display
tab1, tab2, tab3 = st.tabs(["📊 Mean Values (1-10)", "📐 Error Estimates (11-20)", "⚠️ Extreme/Worst Values (21-30)"])

user_inputs = {}

with tab1:
    cols = st.columns(2)
    for i, (name, val) in enumerate(features_info[:10]):
        user_inputs[name] = cols[i % 2].number_input(name, value=float(val), format="%.5f")

with tab2:
    cols = st.columns(2)
    for i, (name, val) in enumerate(features_info[10:20]):
        user_inputs[name] = cols[i % 2].number_input(name, value=float(val), format="%.5f")

with tab3:
    cols = st.columns(2)
    for i, (name, val) in enumerate(features_info[20:]):
        user_inputs[name] = cols[i % 2].number_input(name, value=float(val), format="%.5f")

st.write("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🚀 Run AI Analysis")

if predict_btn:
    try:
        # Convert dictionary values back to 30 float list for FastAPI
        features_list = list(user_inputs.values())
        
        with st.spinner("Analyzing input features..."):
            response = requests.post("http://127.0.0.1:8000/predict", json={"features": features_list})
            
        if response.status_code == 200:
            res = response.json()
            result_text = res.get("result", "")
            
            if "Benign" in result_text:
                st.markdown(f"""
                    <div class="result-card safe">
                        ✅ Prediction: {result_text}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card warning">
                        ⚠️ Prediction: {result_text}
                    </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("📊 API Response Metrics")
            st.json(res)
        else:
            st.error("🚨 FastAPI server unreachable. Make sure uvicorn is running!")
            
    except Exception as e:
        st.error(f"Error processing input: {e}")