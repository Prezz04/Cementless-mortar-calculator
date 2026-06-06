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
        model_path = os.path.join(current_dir, 'xgb_mortar_model.pkl') # Format .pkl baru
        
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat model biner: {str(e)}")
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
        water = st.number_input("Water (ratio by total binders)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

    st.write("---")
    age = st.selectbox("Curing Age (days)", options=[3.0, 7.0, 28.0, 56.0, 91.0], index=2)

    if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
        total_binder = ggbs + cfa + rufa + sf + fa
        
        if water <= 0.0001 or total_binder <= 0.0001:
            st.error("🚨 Kuantitas Air atau komponen Binder tidak boleh nol.")
        else:
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
            
            # Menggunakan struktur DataFrame dengan penamaan kolom yang ketat
            input_df = pd.DataFrame([{
                'GGBS': ggbs, 'CFA': cfa, 'RUFA': rufa, 'SF': sf, 'FA': fa,
                'Aggregate': agg, 'Fiber': fiber, 'SP': sp, 'Age': age,
                'WBR': wbr, 'ABR': abr, 'Log_Age': log_age, 'Sqrt_Age': sqrt_age,
                'SP_x_WBR': sp_x_wbr, 'SP_div_WBR': sp_div_wbr, 
                'GGBS_x_WBR': ggbs_x_wbr, 'FA_x_WBR': fa_x_wbr, 
                'WBR_sq': wbr_sq, 'SP_sq': sp_sq
            }])
            
            # Penyelarasan kolom otomatis berbasis string teks
            kolom_wajib = main_scaler.feature_names_in_
            input_df_final = input_df[kolom_wajib]
            
            scaled_input = main_scaler.transform(input_df_final)
            # Konversi hasil transform ke DataFrame agar model .pkl membaca nama fiturnya
            scaled_input_df = pd.DataFrame(scaled_input, columns=kolom_wajib)
            
            pred_val = max(0.0, xgb_engine.predict(scaled_input_df)[0])
            
            st.success(f"Predicted Compressive Strength: {pred_val:.2f} MPa")
