# 🍜 Hôm Nay Ăn Gì — ML Food Recommender

Dùng Machine Learning (Random Forest) để gợi ý món ăn phù hợp dựa trên sở thích người dùng, kết hợp API bản đồ để hiển thị quán lân cận.

---

## 📁 Cấu trúc project

```
hom-nay-an-gi/
│
├── data/
│   └── food_dataset.csv      ← Dataset 70+ mẫu dữ liệu
│
├── model/                    ← Tự tạo sau khi train
│   ├── food_model.pkl        ← Model đã train
│   ├── encoders.pkl          ← Bộ mã hóa dữ liệu
│   └── metadata.json         ← Thông tin model
│
├── app/
│   └── api.py                ← Flask API server
│
├── train_model.py            ← Script train ML
├── requirements.txt          ← Thư viện cần cài
└── README.md
```

---

## 🚀 Hướng dẫn chạy trên GitHub Codespaces

### Bước 1 — Mở Codespace

1. Vào repo này trên GitHub
2. Nhấn nút **`Code`** (màu xanh)
3. Chọn tab **`Codespaces`**
4. Nhấn **`Create codespace on main`**
5. Chờ ~1 phút → VS Code mở trên browser

---

### Bước 2 — Cài thư viện

Mở **Terminal** trong VS Code (Ctrl + `` ` ``), gõ:

```bash
pip install -r requirements.txt
```

Chờ cài xong (~1-2 phút).

---

### Bước 3 — Train model

```bash
python train_model.py
```

Bạn sẽ thấy output như này:

```
==================================================
  HÔM NAY ĂN GÌ — Training ML Model
==================================================

[1/5] Đọc dataset...
  ✅ Đọc được 72 dòng dữ liệu
  ✅ Có 11 loại món ăn

[2/5] Mã hóa dữ liệu...
  ✅ meal: ['khuya', 'sang', 'toi', 'trua']
  ...

[4/5] Training models...
  🌳 Random Forest Classifier
  ✅ Accuracy test set : 93.33%
  ✅ Cross-val (5-fold): 91.20%

  📊 Feature Importance:
     taste    ████████████████ 0.312
     meal     ████████████ 0.245
     budget   ████████ 0.198
     spicy    █████ 0.142
     group    ███ 0.103

✅ Training hoàn tất!
```

---

### Bước 4 — Chạy API server

```bash
python app/api.py
```

Server chạy tại `http://localhost:5000`

GitHub Codespaces sẽ tự động hỏi bạn có muốn **mở port** không → nhấn **Open in Browser**.

---

### Bước 5 — Test API

Mở terminal thứ 2, gõ lệnh test:

```bash
# Test predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"meal":"trua","budget":"mid","group":"solo","taste":"soup","spicy":"none"}'
```

Kết quả trả về:

```json
{
  "success": true,
  "recommended": "Phở Bò",
  "confidence": 87.3,
  "top3": [
    {"food": "Phở Bò",        "confidence": 87.3},
    {"food": "Bánh Canh Cua", "confidence": 7.2},
    {"food": "Bún Bò Huế",    "confidence": 3.1}
  ],
  "restaurants": [
    {"name": "Phở Hòa Pasteur", "address": "260C Pasteur, Q.3", "distance": "320m", "rating": 4.7}
  ],
  "model_used": "Random Forest"
}
```

---

## 📊 Giá trị hợp lệ cho API

| Trường   | Giá trị hợp lệ |
|----------|----------------|
| `meal`   | `sang` \| `trua` \| `toi` \| `khuya` |
| `budget` | `cheap` \| `mid` \| `high` \| `vip` |
| `group`  | `solo` \| `couple` \| `family` \| `group` |
| `taste`  | `soup` \| `grill` \| `light` \| `fast` |
| `spicy`  | `none` \| `mild` \| `medium` \| `max` |

---

## 🧠 ML Model hoạt động như thế nào?

```
Input người dùng (5 câu hỏi)
        ↓
Label Encoding (chữ → số)
        ↓
Random Forest (100 cây quyết định)
        ↓
Predict_proba → % xác suất từng món
        ↓
Top 1 = Gợi ý chính
Top 3 = Gợi ý thêm
```

**Feature Importance** (mức độ quan trọng):
- `taste` (khẩu vị) — quan trọng nhất ~31%
- `meal`  (bữa ăn)  — ~25%
- `budget` (ngân sách) — ~20%
- `spicy` (độ cay)  — ~14%
- `group` (số người) — ~10%

---

## 🗺️ Tích hợp Google Maps (bước tiếp theo)

Hiện tại dùng dữ liệu quán ăn giả lập. Để dùng bản đồ thật:

1. Đăng ký [Google Maps Platform](https://mapsplatform.google.com/)
2. Lấy API Key
3. Trong `app/api.py`, thay hàm `RESTAURANTS` bằng:

```python
import requests

def get_nearby_restaurants(food_name, lat, lng, api_key):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius"  : 1000,          # 1km
        "type"    : "restaurant",
        "keyword" : food_name,
        "key"     : api_key,
    }
    res = requests.get(url, params=params).json()
    return res["results"][:3]
```

---

## 📈 Cải thiện model (khi có thêm data)

Thêm dòng vào `data/food_dataset.csv` rồi chạy lại `python train_model.py`.

Càng nhiều data → model càng chính xác!

---

## 🛠️ Tech Stack

- **ML**: scikit-learn (Random Forest + KNN)
- **Backend**: Python + Flask
- **Frontend**: HTML/CSS/JS (xem file demo)
- **Map**: Google Maps API (hoặc Goong Maps cho VN)
