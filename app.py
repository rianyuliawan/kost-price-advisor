import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Kost Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_kost_xgb.pkl")
    cols = joblib.load("feature_columns.pkl")
    return model, cols

model, feature_columns = load_model()

# ======================================================
# MODEL ERROR (FROM EVALUATION)
# ======================================================
MODEL_RMSE = 320_000  # sesuaikan dengan RMSE hasil evaluasi kamu

# ======================================================
# DATA REFERENSI AREA (HANYA UNTUK LABEL AREA)
# ======================================================
daftar_kecamatan = [
    "Ciputat", "Ciputat Timur", "Pamulang", "Pondok Aren",
    "Serpong", "Serpong Utara", "Setu",
    "Cisauk", "Curug", "Kelapa Dua", "Pagedangan"
]

# ======================================================
# GLOBAL CSS (FULL SCREEN + NO SCROLL)
# ======================================================
st.markdown("""
<style>
html, body {
    height: 100%;
    overflow: hidden;
    background: radial-gradient(circle at top, #0f172a, #020617 70%);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

section.main > div {
    height: calc(100vh - 70px);
    overflow: hidden;
}

section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

/* HERO */
.hero {
    padding: 32px 36px;
    background: linear-gradient(135deg, #020617, #020617);
    border-radius: 18px;
    box-shadow: 0 0 0 1px #1e293b;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #94a3b8;
    font-size: 15px;
}

/* CARD */
.card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 0 0 1px #1e293b;
    margin-bottom: 20px;
}

/* PRICE */
.price {
    font-size: 46px;
    font-weight: 900;
    color: #22c55e;
}

/* CONFIDENCE BAR */
.conf-bar {
    height: 10px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 10px;
}

.conf-bar-inner {
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #4ade80);
}

/* INSIGHT */
.insight {
    background: #020617;
    border-left: 5px solid #38bdf8;
    padding: 18px;
    border-radius: 12px;
    color: #e0f2fe;
}

/* ANIMATION */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.fade-in {
  animation: fadeUp 0.6s ease-out forwards;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    font-size: 17px;
    font-weight: 700;
    height: 3.3em;
    border-radius: 14px;
    width: 100%;
}

/* TEXT */
.caption {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.title("🏠 Kost Advisor")
    st.caption("AI-based Kost Price Estimation")
    st.markdown("---")
    menu = st.radio("Menu", ["🏠 Estimasi Harga", "ℹ️ Tentang Model"])
    st.markdown("---")
    st.success("Model: XGBoost")

# ======================================================
# ESTIMASI HARGA
# ======================================================
if menu == "🏠 Estimasi Harga":

    st.markdown("""
    <div class="hero">
        <h1>Estimasi Harga Sewa Kost</h1>
        <p>
        Estimasi harga sewa kost berdasarkan lokasi, ukuran kamar,
        dan fasilitas utama menggunakan Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1.9])

    # INPUT
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📍 Lokasi & Ukuran")
        lokasi = st.selectbox("Kecamatan", daftar_kecamatan)
        luas = st.slider(
            "Luas Kamar (m²)",
            min_value=4,
            max_value=20,
            value=12,
            help="Ukuran kamar kost umumnya berada pada rentang 4–20 m²"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✨ Fasilitas Utama")
        ac = st.toggle("❄️ AC")
        km_dalam = st.toggle("🚿 Kamar Mandi Dalam")
        wifi = st.toggle("📶 WiFi")
        kloset = st.toggle("🚽 Kloset Duduk")
        st.markdown('</div>', unsafe_allow_html=True)

        hitung = st.button("💰 Hitung Estimasi Harga")

    # OUTPUT
    with col2:
        if hitung:
            input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
            input_df["Luas_Kamar_m2"] = luas
            input_df["Ada_AC"] = int(ac)
            input_df["Ada_KM_Dalam"] = int(km_dalam)
            input_df["Ada_WiFi"] = int(wifi)
            input_df["Ada_Kloset_Duduk"] = int(kloset)

            loc_col = f"Lokasi_{lokasi}"
            if loc_col in input_df.columns:
                input_df[loc_col] = 1

            # LOADING
            with st.spinner("🤖 Model sedang menganalisis data kost..."):
                time.sleep(1.2)
                pred = int(model.predict(input_df)[0])

            lower = max(pred - MODEL_RMSE, 0)
            upper = pred + MODEL_RMSE

            # RESULT
            st.markdown(f"""
            <div class="card fade-in">
                <h3>Estimasi Harga Bulanan</h3>
                <div class="price">Rp {pred:,}</div>
                <div class="caption">Estimasi berbasis Machine Learning</div>
            </div>
            """, unsafe_allow_html=True)

            # CONFIDENCE RANGE
            st.markdown(f"""
            <div class="card fade-in">
                <b>🔮 Rentang Estimasi Harga</b>
                <p>Rp {lower:,} – Rp {upper:,}</p>
                <div class="conf-bar">
                    <div class="conf-bar-inner" style="width:100%"></div>
                </div>
                <span class="caption">
                    Rentang dihitung berdasarkan rata-rata kesalahan model (RMSE).
                </span>
            </div>
            """, unsafe_allow_html=True)

            # INSIGHT (NON-JUDGMENTAL)
            st.markdown("""
            <div class="insight fade-in">
                <b>💡 Interpretasi Estimasi</b><br>
                Estimasi harga ini mencerminkan nilai sewa kost dengan
                spesifikasi dan fasilitas yang dipilih pada area tersebut.
                Perbedaan harga dapat terjadi antar kost meskipun berada
                di lokasi yang sama, tergantung kondisi dan preferensi penyewa.
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "📌 Estimasi bersifat prediktif dan digunakan sebagai referensi awal, "
                "bukan penentu harga final."
            )

        else:
            st.info("⬅️ Lengkapi data di kiri, lalu klik tombol estimasi.")

# ======================================================
# ABOUT MODEL
# ======================================================
elif menu == "ℹ️ Tentang Model":
    st.header("Tentang Model")
    st.write("""
Model ini menggunakan **XGBoost (Extreme Gradient Boosting)** untuk
mengestimasi harga sewa kost di wilayah Kota Tangerang Selatan.

**Evaluasi Model:**
- Train R² ≈ 70%
- Test R² ≈ 63%
- Selisih Train–Test < 10% (Good Fit)

Model ini digunakan sebagai **estimasi awal harga** dan tidak dimaksudkan
sebagai penentu harga absolut.
""")
