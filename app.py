import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
from streamlit_option_menu import option_menu

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Terapkan CSS
local_css("style.css")

# Sidebar dengan ikon menggunakan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="Aplikasi Prediksi Penyakit Jantung",  
        options=["Dataset", "Preprocessing", "Model"],  
        icons=["database", "tools", "activity"],  
        menu_icon="menu-app",  
        default_index=0,  
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
            },
            "nav-link-selected": {"background-color": "#4C67F5"},
        },
    )

# Halaman Dataset
if selected == "Dataset":
    st.title("Dataset Penyakit Jantung")
    st.write("Sumber Dataset mentah yang digunakan dapat diakses di [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data).")
    st.write("Dataset ini digunakan untuk melatih model prediksi penyakit jantung.")
    st.markdown("""
    **Kolom dalam dataset:**
    - `age`: Usia pasien
    - `sex`: Jenis kelamin pasien (1: Laki-laki, 0: Perempuan)
    - `cp`: Tipe nyeri dada (0-3)
    - `trestbps`: Tekanan darah saat istirahat
    - `chol`: Kolesterol serum
    - `fbs`: Gula darah puasa
    - `restecg`: Hasil elektrokardiografi
    - `thalach`: Detak jantung maksimum
    - `exang`: Angina yang diinduksi olahraga
    - `oldpeak`: Depresi ST
    - `slope`: Kemiringan segmen ST
    - `ca`: Jumlah pembuluh darah berwarna fluoroskopi
    - `thal`: Thalassemia
    - `target`: Indikasi penyakit jantung (1: Ya, 0: Tidak)
    """)
    st.write("Dataset mentah terdapat pada file `heart.csv`.")

# Halaman Preprocessing
elif selected == "Preprocessing":
    st.title("Preprocessing Data")
    st.write("Tahapan preprocessing pada dataset:")
    
    # Tahap 1: Load dataset awal
    st.subheader("1. Dataset Mentah (`heart.csv`)")
    df_raw = pd.read_csv("heart.csv")
    st.write("Data mentah sebelum preprocessing:")
    st.dataframe(df_raw.head())
    st.write(f"Jumlah baris dan kolom: {df_raw.shape}")

    # Tahap 2: Cek dan hapus duplikasi
    st.subheader("2. Hapus Duplikasi (`heart_cleaned.csv`)")
    df_cleaned = df_raw.drop_duplicates()
    st.write("Data setelah duplikasi dihapus:")
    st.dataframe(df_cleaned.head())
    st.write(f"Jumlah baris dan kolom setelah penghapusan duplikasi: {df_cleaned.shape}")

    # Tahap 3: Normalisasi data
    st.subheader("3. Normalisasi Fitur (`heart_scaled.csv`)")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned.drop(columns=["target"])), columns=df_cleaned.drop(columns=["target"]).columns)
    df_scaled["target"] = df_cleaned["target"].values
    st.write("Data setelah normalisasi:")
    st.dataframe(df_scaled.head())
    st.write(f"Jumlah baris dan kolom: {df_scaled.shape}")

    # Simpan hasil preprocessing
    df_cleaned.to_csv("heart_cleaned.csv", index=False)
    df_scaled.to_csv("heart_scaled.csv", index=False)
    st.write("Hasil preprocessing disimpan ke `heart_cleaned.csv` dan `heart_scaled.csv`.")

# Halaman Model
elif selected == "Model":
    st.title("Prediksi Penyakit Jantung")
    st.write("Prediksi yang kami buat menggunakan model terbaik yakni Naive Bayes dengan Seleksi Fitur ANOVA F-value.")
    st.write("Masukkan data di bawah ini untuk memprediksi kemungkinan penyakit jantung.")

    # Muat model, scaler, dan selector
    model = load('naive_bayes_model.joblib')
    scaler = load('scaler.joblib')
    selector = load('selector.joblib')  # Muat selektor fitur

    # Input data dari pengguna
    age = st.number_input("Usia", min_value=0, max_value=120, step=1, value=25)  # Default value 25
    sex = st.selectbox("Jenis Kelamin", ["Pilih...", "Laki-laki (1)", "Perempuan (0)"])
    sex = 1 if sex == "Laki-laki (1)" else 0 if sex == "Perempuan (0)" else None
    cp = st.selectbox("Tipe Nyeri Dada", ["Pilih...", "Tidak ada nyeri dada (0)", "Nyeri dada ringan (1)", "Nyeri dada sedang (2)", "Nyeri dada parah (3)"])
    cp = int(cp.split('(')[-1].split(')')[0]) if cp != "Pilih..." else None
    trestbps = st.number_input("Tekanan Darah Saat Istirahat (mm Hg)", min_value=50, max_value=200, step=1, value=120)
    chol = st.number_input("Kolesterol Serum (mg/dL)", min_value=100, max_value=600, step=1, value=200)
    fbs = st.selectbox("Gula Darah Puasa (> 120 mg/dL)", ["Pilih...", "Tidak (0)", "Ya (1)"])
    fbs = 1 if fbs == "Ya (1)" else 0 if fbs == "Tidak (0)" else None
    restecg = st.selectbox("Hasil Elektrokardiografi", ["Pilih...", "Normal (0)", "Gelombang ST abnormal (1)", "Peningkatan gelombang T (2)"])
    restecg = int(restecg.split('(')[-1].split(')')[0]) if restecg != "Pilih..." else None
    thalach = st.number_input("Detak Jantung Maksimum", min_value=50, max_value=220, step=1, value=150)
    exang = st.selectbox("Angina Induksi Olahraga", ["Pilih...", "Tidak ada angina (0)", "Angina (1)"])
    exang = 1 if exang == "Angina (1)" else 0 if exang == "Tidak ada angina (0)" else None
    oldpeak = st.number_input("Depresi ST (dalam mm)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("Kemiringan Segmen ST", ["Pilih...", "Turun (0)", "Rata (1)", "Naik (2)"])
    slope = int(slope.split('(')[-1].split(')')[0]) if slope != "Pilih..." else None
    ca = st.selectbox("Jumlah Pembuluh Darah Berwarna Fluoroskopi", ["Pilih...", "0 pembuluh (0)", "1 pembuluh (1)", "2 pembuluh (2)", "3 pembuluh (3)", "4 pembuluh (4)"])
    ca = int(ca.split('(')[-1].split(')')[0]) if ca != "Pilih..." else None
    thal = st.selectbox("Thalassemia", ["Pilih...", "Tidak (0)", "Defisiensi (1)", "Penyakit Thalassemia (2)", "Mild (3)"])
    thal = int(thal.split('(')[-1].split(')')[0]) if thal != "Pilih..." else None

    # Validasi input
    if st.button("Prediksi"):
        # Periksa apakah ada input yang kosong atau tidak valid
        if (
            sex is None
            or cp is None
            or fbs is None
            or restecg is None
            or exang is None
            or slope is None
            or ca is None
            or thal is None
        ):
            st.error("Harap isi semua Data dengan benar sebelum melakukan prediksi!")
        else:
            # Proses input ke DataFrame
            user_input = pd.DataFrame(
                [[
                    age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal
                ]],
                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            )

            # Preprocessing dengan scaler
            try:
                user_input_scaled = scaler.transform(user_input)
            except ValueError as e:
                st.error(f"Error pada proses skala data: {e}")
                st.stop()

            # Seleksi fitur dengan selector
            try:
                user_input_selected = selector.transform(user_input_scaled)
            except ValueError as e:
                st.error(f"Error pada seleksi fitur: {e}")
                st.stop()

            # Prediksi
            prediction = model.predict(user_input_selected)
            probability = model.predict_proba(user_input_selected)

            # Hasil prediksi
            st.write(f"Jenis Kelamin: {sex}, Tipe Nyeri Dada: {cp}, Gula Darah Puasa: {fbs}, Hasil Elektrokardiografi: {restecg}, Angina Induksi Olahraga: {exang}, Kemiringan Segmen ST: {slope}, Jumlah Pembuluh Darah: {ca}, Thalassemia: {thal}")
            st.write(f"**Hasil Prediksi:** {'Diagnosa Penyakit Jantung' if prediction[1] == 0 else 'Tidak Mengidap Penyakit Jantung'}")
            st.write(f"**Probabilitas:** {probability[0][1]*100:.2f}% kemungkinan penyakit jantung")
