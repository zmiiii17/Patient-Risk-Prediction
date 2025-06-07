import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Risiko Pasien", layout="centered")
st.title("🎯 Prediksi Risiko Pasien Menggunakan Decision Tree")
st.caption("Dataset: Rumah Sakit XYZ | Fitur: Gaya Hidup, Kondisi Pasien")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("predic table.csv")
    df = df.drop("No", axis=1)  # kolom No tidak dipakai
    return df

data = load_data()
st.subheader("📊 Data Pasien")
st.dataframe(data.head())

# Encoding fitur kategorikal
label_encoders = {}
df_encoded = data.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

# Split data
X = df_encoded.drop("Hasil", axis=1)
y = df_encoded["Hasil"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Akurasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"🎉 Akurasi Model: {acc*100:.2f}%")

# Visualisasi Decision Tree
st.subheader("🌳 Visualisasi Pohon Keputusan")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=["Tidak", "Ya"], filled=True, ax=ax)
st.pyplot(fig)

# Form untuk prediksi manual
st.subheader("🔍 Prediksi Risiko Baru")

def user_input():
    usia = st.selectbox("Usia", data["Usia"].unique())
    jk = st.selectbox("Jenis Kelamin", data["Jenis_Kelamin"].unique())
    merokok = st.selectbox("Merokok", data["Merokok"].unique())
    bekerja = st.selectbox("Bekerja", data["Bekerja"].unique())
    rumah = st.selectbox("Rumah Tangga", data["Rumah_Tangga"].unique())
    begadang = st.selectbox("Aktivitas Begadang", data["Aktivitas_Begadang"].unique())
    olahraga = st.selectbox("Aktivitas Olahraga", data["Aktivitas_Olahraga"].unique())
    asuransi = st.selectbox("Asuransi", data["Asuransi"].unique())
    penyakit = st.selectbox("Penyakit Bawaan", data["Penyakit_Bawaan"].unique())

    df = pd.DataFrame([[usia, jk, merokok, bekerja, rumah, begadang, olahraga, asuransi, penyakit]],
                      columns=X.columns)

    # encode input
    for col in df.columns:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    return df

input_df = user_input()
pred = model.predict(input_df)[0]
hasil_label = label_encoders["Hasil"].inverse_transform([pred])[0]

st.info(f"🧾 Prediksi Risiko Pasien: **{hasil_label}**")

# Tabel data + prediksi (optional)
st.subheader("📋 Hasil Prediksi Data Asli")
df_encoded["Prediksi"] = model.predict(X)
df_encoded["Prediksi_Label"] = label_encoders["Hasil"].inverse_transform(df_encoded["Prediksi"])
st.dataframe(df_encoded)
