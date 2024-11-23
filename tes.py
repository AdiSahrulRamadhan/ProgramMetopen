import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump

# Load dataset
df_heart = pd.read_csv('heart_scaled.csv')  # Ganti dengan nama file cleaned Anda

# Pisahkan fitur (X) dan target (y)
X = df_heart.drop(columns=['target'])  # Pastikan kolom 'target' adalah label
y = df_heart['target']

# Tangani missing value menggunakan SimpleImputer (strategi median)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# **1. Seleksi Fitur menggunakan ANOVA F-value (Metode Filter)**
selector = SelectKBest(score_func=f_classif, k=10)  # Pilih 10 fitur terbaik
X_new = selector.fit_transform(X_scaled, y)
selected_features = [X.columns[i] for i in range(len(X.columns)) if selector.get_support()[i]]
print(f"Fitur yang terpilih menggunakan ANOVA F-value: {selected_features}")

# Membagi dataset menjadi data latih dan data uji
X_combined = pd.DataFrame(X_scaled, columns=X.columns)[selected_features].values
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=123)

# Membuat model Naive Bayes Gaussian
gnb_model = GaussianNB()

# Melatih model menggunakan data latih
gnb_model.fit(X_train, y_train)

# Memprediksi label untuk data uji
y_pred = gnb_model.predict(X_test)

# Mengevaluasi performa model
accuracy = accuracy_score(y_pred, y_test) * 100
print(f'Akurasi Naive Bayes dengan seleksi fitur ANOVA F-value: {accuracy:.2f}%')

# Simpan model dan scaler
dump(gnb_model, 'naive_bayes_model.joblib')
dump(scaler, 'scaler.joblib')
dump(selector, 'selector.joblib')  # Simpan seleksi fitur
print("Model, scaler, dan seleksi fitur berhasil disimpan!")
