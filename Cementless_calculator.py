import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE GUI (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Concrete Strength Calculator", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size:28px; font-weight:bold; text-align:center; color:#1E293B; margin-bottom:20px; }
    
    /* Frame Tunggal Menggunakan Tema Hijau Telur Asin (Soft Teal) */
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
# 2. LOAD BRAIN ENGINE (Membaca File Biner Mandiri)
# ==========================================
@st.cache_resource
def load_prediction_engine():
    try:
        # Membaca paket model biner secara langsung (Bukan dengan pd.read_csv)
        artifacts = joblib.load('xgb_mortar_model.pkl')
        return artifacts['model'], artifacts['scaler'], artifacts['columns']
    except FileNotFoundError:
        st.error("Error: File 'xgb_mortar_model.pkl' tidak ditemukan di repositori GitHub Anda.")
        return None, None, None

xgb_engine, main_scaler, feature_columns = load_prediction_engine()

# Fungsi pembentuk fitur modular untuk input user
def feature_extractor(data_df):
    df_res = data_df.copy()
    binder_cols = ['GGBS', 'CFA', 'RUFA', 'SF', 'FA']
    df_res['Total_Binder'] = df_res[binder_cols].sum(axis=1)
    safe_binder = df_res['Total_Binder'].replace(0, 1e-6)
    
    df_res['WBR'] = df_res['Water'] / safe_binder
    df_res['ABR'] = df_res['Aggregate'] / safe_binder
    df_res['Log_Age'] = np.log1p(df_res['Age'])
    df_res['Sqrt_Age'] = np.sqrt(df_res['Age'])
    df_res['SP_x_WBR'] = df_res['SP'] * df_res['WBR']
    df_res['SP_div_WBR'] = df_res.apply(lambda r: r['SP'] / r['WBR'] if r['WBR'] > 0 else 0, axis=1)
    df_res['GGBS_x_WBR'] = df_res['GGBS'] * df_res['WBR']
    df_res['FA_x_WBR'] = df_res['FA'] * df_res['WBR']
    df_res['WBR_sq'] = df_res['WBR']**2
    df_res['SP_sq'] = df_res['SP']**2
    return df_res

# ==========================================
# 3. INTERFACE PENGGUNA INTERAKTIF (FRONTEND)
# ==========================================
if xgb_engine is not None:
    st.write("### Input Mix Design Parameters:")
    
    # Grid susunan input 3 kolom simetris
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
    
    # Mengunci variasi umur pada rentang data laboratorium aktual untuk menjaga logika kestabilan fisik pohon keputusan
    age = st.selectbox("Curing Age (days)", options=[3.0, 7.0, 28.0, 56.0, 91.0], index=2)

    # Logika Tombol Eksekusi Prediksi
    if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
        total_binder = ggbs + cfa + rufa + sf + fa
        
        # Sensor Keamanan Fisika Mortar
        if water <= 0.0001 or total_binder <= 0.0001:
            st.error("🚨 INVALID MIX DESIGN! Kuantitas Air atau komponen Binder tidak boleh nol.")
        else:
            # Mengubah data input user menjadi kerangka DataFrame
            input_df = pd.DataFrame([{
                'GGBS': ggbs, 'CFA': cfa, 'RUFA': rufa, 'SF': sf, 'FA': fa,
                'Aggregate': agg, 'Fiber': fiber, 'SP': sp, 'Water': water, 'Age': age
            }])
            
            # Ekstraksi fitur, alignment kolom, dan normalisasi skala
            processed_input = feature_extractor(input_df)
            processed_input = processed_input[feature_columns]
            scaled_input = main_scaler.transform(processed_input)
            
            # Prediksi Titik Utama Kuat Tekan
            pred_val = xgb_engine.predict(scaled_input)[0]
            
            # Perhitungan Batas Ketidakpastian Diperluas ke 95% PI (Z-Score: 1.96)
            mae_calibration = 1.64
            uncertainty_margin = mae_calibration * 1.96
            
            lower_bound = max(0.0, pred_val - uncertainty_margin)
            upper_bound = pred_val + uncertainty_margin

            # Rendering Output dalam SATU FRAME UNTUK JURNAL
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
