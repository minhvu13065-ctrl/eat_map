"""
==============================================
  HÔM NAY ĂN GÌ — Train ML Model
  Dùng: Random Forest Classifier
  Chạy: python train_model.py
==============================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os

print("=" * 50)
print("  HÔM NAY ĂN GÌ — Training ML Model")
print("=" * 50)

# ─────────────────────────────────────────────
# 1. ĐỌC DATA
# ─────────────────────────────────────────────
print("\n[1/5] Đọc dataset...")
df = pd.read_csv("data/food_dataset.csv")
print(f"  ✅ Đọc được {len(df)} dòng dữ liệu")
print(f"  ✅ Có {df['food_label'].nunique()} loại món ăn:")
for food in sorted(df['food_label'].unique()):
    count = len(df[df['food_label'] == food])
    print(f"     • {food}: {count} mẫu")

# ─────────────────────────────────────────────
# 2. ENCODE FEATURES
# ─────────────────────────────────────────────
print("\n[2/5] Mã hóa dữ liệu (Label Encoding)...")

features = ['meal', 'budget', 'group', 'taste', 'spicy']
target   = 'food_label'

encoders = {}
df_encoded = df.copy()

for col in features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  ✅ {col}: {list(le.classes_)}")

# Encode nhãn đầu ra
label_encoder = LabelEncoder()
df_encoded[target] = label_encoder.fit_transform(df[target])
encoders['food_label'] = label_encoder
print(f"  ✅ food_label: {list(label_encoder.classes_)}")

# ─────────────────────────────────────────────
# 3. CHIA DATA TRAIN / TEST
# ─────────────────────────────────────────────
print("\n[3/5] Chia tập train / test (80% / 20%)...")

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✅ Train: {len(X_train)} mẫu")
print(f"  ✅ Test:  {len(X_test)} mẫu")

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[4/5] Training models...")

# --- Model 1: Random Forest ---
print("\n  🌳 Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 cây quyết định
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)
rf_cv    = cross_val_score(rf_model, X, y, cv=5).mean()
print(f"  ✅ Accuracy test set : {rf_acc:.2%}")
print(f"  ✅ Cross-val (5-fold): {rf_cv:.2%}")

# Feature Importance
importances = rf_model.feature_importances_
print("\n  📊 Feature Importance (tầm quan trọng):")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"     {feat:<8} {bar} {imp:.3f}")

# --- Model 2: KNN ---
print("\n  🔵 K-Nearest Neighbors (KNN, k=5)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc  = accuracy_score(y_test, knn_pred)
knn_cv   = cross_val_score(knn_model, X, y, cv=5).mean()
print(f"  ✅ Accuracy test set : {knn_acc:.2%}")
print(f"  ✅ Cross-val (5-fold): {knn_cv:.2%}")

# Chọn model tốt hơn
best_model      = rf_model if rf_cv >= knn_cv else knn_model
best_model_name = "Random Forest" if rf_cv >= knn_cv else "KNN"
print(f"\n  🏆 Model được chọn: {best_model_name}")

# ─────────────────────────────────────────────
# 5. LƯU MODEL
# ─────────────────────────────────────────────
print("\n[5/5] Lưu model...")

os.makedirs("model", exist_ok=True)

# Lưu model
with open("model/food_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Lưu encoders
with open("model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Lưu metadata (dùng cho frontend)
metadata = {
    "model_name"   : best_model_name,
    "accuracy"     : round(rf_cv, 4),
    "features"     : features,
    "foods"        : list(label_encoder.classes_),
    "feature_values": {
        col: list(encoders[col].classes_)
        for col in features
    }
}
with open("model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("  ✅ model/food_model.pkl")
print("  ✅ model/encoders.pkl")
print("  ✅ model/metadata.json")

# ─────────────────────────────────────────────
# TEST PREDICT THỬ
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  THỬ PREDICT")
print("=" * 50)

def predict_food(meal, budget, group, taste, spicy):
    """Dự đoán món ăn từ input người dùng"""
    input_data = {}
    for col, val in zip(features, [meal, budget, group, taste, spicy]):
        input_data[col] = encoders[col].transform([val])[0]
    
    X_input = pd.DataFrame([input_data])
    
    # Dự đoán + xác suất
    pred_idx   = best_model.predict(X_input)[0]
    pred_proba = best_model.predict_proba(X_input)[0]
    pred_food  = label_encoder.inverse_transform([pred_idx])[0]
    
    # Top 3
    top3_idx   = np.argsort(pred_proba)[::-1][:3]
    top3       = [(label_encoder.inverse_transform([i])[0], round(pred_proba[i]*100, 1))
                  for i in top3_idx]
    
    return pred_food, top3

# Ví dụ 1
food, top3 = predict_food("trua", "mid", "solo", "soup", "none")
print(f"\n  Input : Trưa | Mid | Solo | Soup | Không cay")
print(f"  Kết quả: {food}")
print(f"  Top 3 : {top3}")

# Ví dụ 2
food2, top3_2 = predict_food("toi", "high", "group", "grill", "medium")
print(f"\n  Input : Tối | High | Nhóm | Nướng | Cay vừa")
print(f"  Kết quả: {food2}")
print(f"  Top 3 : {top3_2}")

print("\n✅ Training hoàn tất! File model đã lưu trong /model/")
print("👉 Bước tiếp: chạy  python app/api.py  để khởi động API server")
