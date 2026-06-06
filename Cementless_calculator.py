import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE GUI
# ==========================================
st.set_page_config(page_title="Concrete Strength Calculator", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size:28px; font-weight:bold; text-align:center; color:#1E293B; margin-bottom:20px; }
    .unified-output-box { 
        background-color: #F0FDFA; 
        border-left: 6px solid #0D9488; 
        padding: 20px; 
        border-radius: 8px; 
        margin-top: 25px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .unified-prediction { font-size: 22px; font-weight: bold; color: #115E59; margin-bottom: 12px; }
    .unified-divider { border-top: 1px solid #99F6E4; margin: 12px 0; }
    .unified-title { font-size: 16px; font-weight: bold; color: #115E59; }
    .unified-text { font-size: 14px; color: #134E4A; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Prediction of Compressive Strength in Cementless Mortar</div>', unsafe_allow_html=True)

# ==========================================
# 2. LOAD BRAIN ENGINE (Path Absolut)
# ==========================================
@st.cache_resource
def load_prediction_engine():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(current_dir, 'scaler_mortar.pkl')
        model_path = os.path.join(current_dir, 'xgb_mortar_model.json')
        
        scaler = joblib.load(scaler_path)
        
        model = XGBRegressor()
        model.load_model(model_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat model biner: {str(e)}")
        return None, None

xgb_engine, main_scaler = load_prediction_engine()

# ==========================================
# 3. INTERFACE PENGGUNA INTERAKTIF (FRONTEND)
# ==========================================
if xgb_engine is not None:
    st.write("### Input Mix Design Parameters:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ggbs = st.number_input("GGBS (ratio)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        sf = st.number_input("Silica Fume (ratio)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        fiber = st.number_input("Fiber (ratio)", min_value=0.0, max_value=1.0, value=0.005, step=0.001, format="%.3f")

    with col2:
        cfa = st.number_input("Co-fired Fly Ash (ratio)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        fa = st.number_input("Fly Ash (ratio)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        sp = st.number_input("Superplasticizer (ratio)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)

    with col3:
        rufa = st.number_input("Reactive Ultrafine FA (ratio)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        agg = st.number_input("Fine Aggregate (ratio by total binders)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        water = st.number_input("Water (ratio by total binders)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

    st.write("---")
    age = st.selectbox("Curing Age (days)", options=[3.0, 7.0, 28.0, 56.0, 91.0], index=2)

    if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
        total_binder = ggbs + cfa + rufa + sf + fa
        
        if water <= 0.0001 or total_binder <= 0.0001:
            st.error("🚨 INVALID MIX DESIGN! Kuantitas Air atau komponen Binder tidak boleh nol.")
        else:
            # Kalkulasi matematika manual untuk Advanced Feature Engineering
            safe_binder = total_binder if total_binder > 0 else 1e-6
            wbr = water / safe_binder
            abr = agg / safe_binder
            log_age = np.log1p(age)
            sqrt_age = np.sqrt(age)
            sp_x_wbr = sp * wbr
            sp_div_wbr = sp / wbr if wbr > 0 else 0
            ggbs_x_wbr = ggbs * wbr
            fa_x_wbr = fa * wbr
            wbr_sq = wbr ** 2
            sp_sq = sp ** 2
            
            # SUSUN ARRAY 18 FITUR PERSIS SESUAI URUTAN FITUR TRAINING NOTEBOOK JUPYTER
            # Urutan: 8 Fitur Dasar -> 10 Fitur Hasil Rekayasa Matematika
            raw_features = np.array([[
                ggbs, cfa, rufa, sf, fa, agg, fiber, sp,
                wbr, abr, log_age, sqrt_age, sp_x_wbr, sp_div_wbr,
                ggbs_x_wbr, fa_x_wbr, wbr_sq, sp_sq
            ]])
            
            # Mematikan validasi string nama kolom scikit-learn secara paksa
            if hasattr(main_scaler, 'feature_names_in_'):
                del main_scaler.feature_names_in_
                
            # Eksekusi penskalaan berbasis data array murni
            scaled_input = main_scaler.transform(raw_features)
            
            # Eksekusi prediksi model final
            raw_pred = xgb_engine.predict(scaled_input)[0]
            pred_val = max(0.0, raw_pred)  # Batas fisik minimum beton = 0 MPa
            
            # Kalibrasi Ketidakpastian 95% PI
            mae_calibration = 1.64
            uncertainty_margin = mae_calibration * 1.96
            lower_bound = max(0.0, pred_val - uncertainty_margin)
            upper_bound = pred_val + uncertainty_margin

            st.markdown(f"""
                <div class="unified-output-box">
                    <div class="unified-prediction">
                        Predicted Compressive Strength: {pred_val:.2f} MPa
                    </div>
                    <div class="unified-divider"></div>
                    <div class="unified-title">
                        Reliability Analysis (95% Predictive Interval)
                    </div>
                    <div class="unified-text">
                        Based on XGBoost historical residual calibration (MAE: 1.64 MPa), the statistical boundaries for this specific alternative mix configuration fall within:<br>
                        <strong>[{lower_bound:.2f} MPa — {upper_bound:.2f} MPa]</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
