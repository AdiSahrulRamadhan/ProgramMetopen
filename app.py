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
    age = st.number_input("Usia", min_value=0, max_value=120, step=1, value=25, format="%d")  # Default value 25
    st.text("Usia pasien dalam tahun.")

    sex = st.selectbox("Jenis Kelamin", ["Pilih...", "Laki-laki", "Perempuan"])
    st.text("Jenis kelamin pasien. Pilih antara Laki-laki atau Perempuan.")

    cp = st.selectbox("Tipe Nyeri Dada", ["Pilih...", 0, 1, 2, 3])
    st.text("Kode untuk tipe nyeri dada: 0: Tidak ada, 1: Tipe 1, 2: Tipe 2, 3: Tipe 3.")

    trestbps = st.number_input("Tekanan Darah Saat Istirahat (mm Hg)", min_value=50, max_value=200, step=1, value=120)
    st.text("Tekanan darah saat istirahat dalam satuan mm Hg.")

    chol = st.number_input("Kolesterol Serum (mg/dL)", min_value=100, max_value=600, step=1, value=200)
    st.text("Kolesterol serum dalam satuan mg/dL.")

    fbs = st.selectbox("Gula Darah Puasa (> 120 mg/dL)", ["Pilih...", 0, 1])
    st.text("Status gula darah puasa: 0: Normal, 1: Lebih dari 120 mg/dL.")

    restecg = st.selectbox("Hasil Elektrokardiografi", ["Pilih...", 0, 1, 2])
    st.text("Hasil elektrokardiografi: 0: Normal, 1: ST-T abnormal, 2: Hipertrofi ventrikel.")

    thalach = st.number_input("Detak Jantung Maksimum", min_value=50, max_value=220, step=1, value=150)
    st.text("Detak jantung maksimum yang tercatat.")

    exang = st.selectbox("Angina Induksi Olahraga", ["Pilih...", 0, 1])
    st.text("Indikasi angina akibat olahraga: 0: Tidak, 1: Ya.")

    oldpeak = st.number_input("Depresi ST (dalam mm)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    st.text("Nilai depresi segmen ST pada saat tes olahraga.")

    slope = st.selectbox("Kemiringan Segmen ST", ["Pilih...", 0, 1, 2])
    st.text("Kemiringan segmen ST setelah tes olahraga.")

    ca = st.selectbox("Jumlah Pembuluh Darah Berwarna Fluoroskopi", ["Pilih...", 0, 1, 2, 3, 4])
    st.text("Jumlah pembuluh darah berwarna pada fluoroskopi.")

    thal = st.selectbox("Thalassemia", ["Pilih...", 0, 1, 2, 3])
    st.text("Status thalassemia: 0: Tidak, 1: Fix defect, 2: Reversable defect, 3: Normal.")

    # Validasi input
    if st.button("Prediksi"):
        # Periksa apakah ada input yang kosong atau tidak valid
        if (
            sex == "Pilih..."
            or cp == "Pilih..."
            or fbs == "Pilih..."
            or restecg == "Pilih..."
            or exang == "Pilih..."
            or slope == "Pilih..."
            or ca == "Pilih..."
            or thal == "Pilih..."
        ):
            st.error("Harap isi semua Data dengan benar sebelum melakukan prediksi!")
        else:
            # Proses input ke DataFrame
            user_input = pd.DataFrame(
                [[
                    age, 1 if sex == "Laki-laki" else 0, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal
                ]],
                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            )

            # Scaling input dan seleksi fitur
            user_input_scaled = scaler.transform(user_input)
            user_input_selected = selector.transform(user_input_scaled)

            # Prediksi hasil
            prediction = model.predict(user_input_selected)
            prediction_prob = model.predict_proba(user_input_selected)[0][1]

            # Tampilkan hasil prediksi
            if prediction[0] == 1:
                st.success(f"Prediksi: Penyakit jantung terdeteksi dengan probabilitas {prediction_prob:.2f}")
            else:
                st.success(f"Prediksi: Tidak ada penyakit jantung dengan probabilitas {prediction_prob:.2f}")
