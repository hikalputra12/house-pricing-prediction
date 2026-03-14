from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import joblib
import uvicorn

#Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="House Price Prediction API", 
    description="API Inference untuk prediksi rumah menggunakan Gradient Boosting",
    version="1.0"
)

# Muat Artifacts (Dijalankan sekali saat server menyala)
try:
    model = joblib.load('/home/haikal/Downloads/house-pricing/proyek_house_prediction_python/model/model_gbr_final.joblib') 
    preprocessor = joblib.load('/home/haikal/Downloads/house-pricing/proyek_house_prediction_python/model/preprocessor.joblib')
    print("✅ Model dan Preprocessor berhasil dimuat ke memori!")
except Exception as e:
    print(f"❌ Gagal memuat file joblib: {e}")

# Endpoint Prediksi
@app.post("/predict")
def predict_price(data: Dict[str, Any]):
    """
    Endpoint ini menerima JSON berisi 76 fitur rumah mentah,
    memprosesnya, dan mengembalikan prediksi harga asli.
    """
    try:
        # Konversi: JSON (Dictionary) -> Pandas DataFrame (1 Baris)
        df_input = pd.DataFrame([data])
        
        # Transformasi: Ubah data mentah ke matriks yang dipahami model
        # Preprocessor otomatis menangani One-Hot Encoding teks & Scaling angka
        X_processed = preprocessor.transform(df_input)
        
        # Prediksi: Model menebak harga
        # Karena target (y) tidak distandarisasi saat training, 
        # hasilnya sudah langsung berupa harga asli!
        prediction = model.predict(X_processed)
        #responnya
        return {
            "status": "success",
            "predicted_price": float(prediction[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan pemrosesan data: {str(e)}")

#menjalankan file ini langsung dengan `python main.py`
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)