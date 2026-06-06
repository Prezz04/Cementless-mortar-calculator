import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Concrete Strength Calculator", layout="centered")

@st.cache_resource
def load_prediction_engine():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(current_dir, 'scaler_mortar.pkl')
        model_path = os.path.join(current_dir, 'xgb_mortar_model.pkl')
        
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat file biner: {str(e)}")
        return None, None

xgb_engine, main_scaler = load_prediction_engine()

if xgb_engine is not None:
    st.write("### Input Mix Design Parameters:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ggbs = st.number_input("GGBS (ratio)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        sf = st.number_input("Silica Fume (ratio)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        fiber = st.number_input("Fiber (ratio)", min_value=0.0, max_value=1.0, value=0.005, step=0.001, format="%.3f")

    with col2:
        cfa = st.number_input("Co-fired Fly Ash (ratio)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        fa = st.number_input("Fly Ash (ratio)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        sp = st.number_input("Superplasticizer (ratio)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)

    with col3:
        rufa = st.number_input("Reactive Ultrafine FA (ratio)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        agg = st.number_input("Fine Aggregate (ratio by total binders)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        # Form Water tetap ada sebagai interface, tapi tidak ikut dikirim ke model jika model lu cuma minta 9 kolom dasar
        water = st.number_input("Water (ratio by total binders)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

    st.write("---")
    age = st.selectbox("Curing Age (days)", options=[3.0, 7.0, 28.0, 56.0, 91.0], index=2)

    if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
        # Langsung susun 9 parameter asli tanpa rekayasa rumus yang berisiko beda formula
        input_df = pd.DataFrame([{
            'GGBS': ggbs, 'CFA': cfa, 'RUFA': rufa, 'SF': sf, 'FA': fa,
            'Aggregate': agg, 'Fiber': fiber, 'SP': sp, 'Age': age
        }])
        
        kolom_wajib = list(main_scaler.feature_names_in_)
        input_df_final = input_df[kolom_wajib]
        
        scaled_array = main_scaler.transform(input_df_final)
        scaled_df_final = pd.DataFrame(scaled_array, columns=kolom_wajib)
        
        pred_val = max(0.0, float(xgb_engine.predict(scaled_df_final)[0]))
        
        st.success(f"Predicted Compressive Strength: {pred_val:.2f} MPa")
