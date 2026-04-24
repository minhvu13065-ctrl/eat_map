"""
==============================================
  HÔM NAY ĂN GÌ — Flask API Server
  Chạy: python app/api.py
  API:  POST /predict
==============================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

# ── Load model + encoders ──────────────────
print("Loading model...")
with open("model/food_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("model/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

features      = metadata["features"]
label_encoder = encoders["food_label"]
print(f"✅ Model loaded: {metadata['model_name']} (accuracy: {metadata['accuracy']:.2%})")

# ── Dữ liệu quán ăn giả lập (sau thay bằng Google Maps API) ──
RESTAURANTS = {
    "Phở Bò": [
        {"name": "Phở Hòa Pasteur",    "address": "260C Pasteur, Q.3",          "distance": "320m",  "rating": 4.7, "lat": 10.7769, "lng": 106.6927},
        {"name": "Phở 24 Nguyễn Trãi", "address": "12 Nguyễn Trãi, Q.1",        "distance": "680m",  "rating": 4.5, "lat": 10.7725, "lng": 106.6980},
        {"name": "Phở Lệ",             "address": "413 Nguyễn Trãi, Q.5",        "distance": "1.2km", "rating": 4.6, "lat": 10.7531, "lng": 106.6669},
    ],
    "Bánh Mì": [
        {"name": "Bánh Mì Huỳnh Hoa",  "address": "26 Lê Thị Riêng, Q.1",       "distance": "250m",  "rating": 4.8, "lat": 10.7800, "lng": 106.6950},
        {"name": "Bánh Mì 37",         "address": "37 Nguyễn Trãi, Q.1",        "distance": "540m",  "rating": 4.5, "lat": 10.7740, "lng": 106.6970},
        {"name": "Bánh Mì Phượng",     "address": "80 Lê Lợi, Q.1",            "distance": "900m",  "rating": 4.4, "lat": 10.7730, "lng": 106.7020},
    ],
    "Cơm Tấm": [
        {"name": "Cơm Tấm Thuận Kiều", "address": "129 Hùng Vương, Q.5",        "distance": "350m",  "rating": 4.7, "lat": 10.7550, "lng": 106.6800},
        {"name": "Cơm Tấm 94",         "address": "94 Trần Văn Đang, Q.3",      "distance": "620m",  "rating": 4.5, "lat": 10.7760, "lng": 106.6900},
        {"name": "Cơm Tấm Mộc",        "address": "234 Cách Mạng Tháng 8, Q.3", "distance": "980m",  "rating": 4.4, "lat": 10.7800, "lng": 106.6860},
    ],
}
DEFAULT_RESTAURANTS = [
    {"name": "Quán Ngon 138",    "address": "138 Nam Kỳ Khởi Nghĩa, Q.3", "distance": "280m",  "rating": 4.6, "lat": 10.7800, "lng": 106.6930},
    {"name": "Bếp Gia Đình",    "address": "45 Đinh Tiên Hoàng, Q.BT",   "distance": "650m",  "rating": 4.4, "lat": 10.8010, "lng": 106.7050},
    {"name": "Nhà Hàng Phú Gia","address": "22 Hoàng Diệu, Q.4",         "distance": "1.0km", "rating": 4.5, "lat": 10.7620, "lng": 106.7050},
]

# ──────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Hôm Nay Ăn Gì API 🍜",
        "model"  : metadata["model_name"],
        "accuracy": metadata["accuracy"],
        "endpoints": {
            "POST /predict": "Dự đoán món ăn",
            "GET  /foods"  : "Danh sách món ăn",
            "GET  /health" : "Kiểm tra server",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/foods", methods=["GET"])
def get_foods():
    return jsonify({"foods": metadata["foods"]})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Input JSON:
    {
        "meal"  : "trua",    // sang | trua | toi | khuya
        "budget": "mid",     // cheap | mid | high | vip
        "group" : "solo",    // solo | couple | family | group
        "taste" : "soup",    // soup | grill | light | fast
        "spicy" : "none"     // none | mild | medium | max
    }
    """
    try:
        data = request.get_json()

        # Validate input
        required = ["meal", "budget", "group", "taste", "spicy"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Thiếu trường: {field}"}), 400

        # Encode input
        encoded = {}
        for col in features:
            val = data[col]
            valid_vals = list(encoders[col].classes_)
            if val not in valid_vals:
                return jsonify({"error": f"Giá trị '{val}' không hợp lệ cho '{col}'. Chọn: {valid_vals}"}), 400
            encoded[col] = encoders[col].transform([val])[0]

        X_input = pd.DataFrame([encoded])

        # Predict
        pred_idx   = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]
        pred_food  = label_encoder.inverse_transform([pred_idx])[0]

        # Top 3 gợi ý
        top3_idx = np.argsort(pred_proba)[::-1][:3]
        top3 = [
            {
                "food"      : label_encoder.inverse_transform([i])[0],
                "confidence": round(float(pred_proba[i]) * 100, 1)
            }
            for i in top3_idx
        ]

        # Lấy quán lân cận
        restaurants = RESTAURANTS.get(pred_food, DEFAULT_RESTAURANTS)

        return jsonify({
            "success"    : True,
            "recommended": pred_food,
            "confidence" : round(float(pred_proba[pred_idx]) * 100, 1),
            "top3"       : top3,
            "restaurants": restaurants,
            "model_used" : metadata["model_name"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 Server chạy tại: http://localhost:5000")
    print("📡 Test API: curl -X POST http://localhost:5000/predict \\")
    print('   -H "Content-Type: application/json" \\')
    print('   -d \'{"meal":"trua","budget":"mid","group":"solo","taste":"soup","spicy":"none"}\'')
    app.run(debug=True, port=5000)
