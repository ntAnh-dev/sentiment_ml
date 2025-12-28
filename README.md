# Sentiment Analysis ML (Local MLOps with Git + DVC + Jenkins)

Dự án này minh họa **một pipeline MLOps chạy hoàn toàn local** cho bài toán **phân loại cảm xúc câu nói (Sentiment Analysis)**, sử dụng **Machine Learning cổ điển (không deep learning)**, kết hợp:

* **Git**: quản lý source code
* **DVC**: quản lý dữ liệu + pipeline train
* **MLflow**: tracking & so sánh mô hình
* **Docker / Docker Compose**: deploy API
* **Jenkins (local)**: CI/CD tự động train & deploy khi data/code thay đổi

> 🎯 Mục tiêu: Chỉ cần thay đổi **data hoặc code ở local**, Jenkins sẽ **tự động phát hiện → train lại → deploy API** mà **không cần push Git hay DVC**.

---

## 1. Bài toán

* **Input**: câu nói / bình luận (text)
* **Output**: nhãn cảm xúc (positive / negative)
* **Loại bài toán**: Text Classification

---

## 2. Các mô hình ML sử dụng

Trong `train_compare.py`, hệ thống tự động train & so sánh **3 mô hình ML cổ điển**:

1. **Logistic Regression + TF-IDF**
2. **Linear SVM + TF-IDF**
3. **Multinomial Naive Bayes + TF-IDF**

### Metric đánh giá

* Accuracy
* Precision
* Recall
* F1-score

➡️ Mô hình tốt nhất sẽ được lưu thành:

```
models/best_model.joblib
```

và log đầy đủ vào **MLflow**.

---

## 3. Cấu trúc thư mục

```
sentiment_ml/
├── src/
│   ├── train_compare.py     # Train + compare 3 mô hình
│   ├── api.py               # FastAPI serve model
│   └── utils.py
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── best_model.joblib    # Model được deploy
│
├── tests/
│   └── test_training.py     # Unit test
│
├── dvc.yaml                 # DVC pipeline
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.api
├── Jenkinsfile (optional)
├── requirements.txt
└── README.md
```

---

## 4. DVC pipeline

### dvc.yaml

```yaml
stages:
  train:
    cmd: python src/train_compare.py
    deps:
      - src/train_compare.py
      - data/train.csv
      - data/test.csv
    outs:
      - models/best_model.joblib
```

### Ý nghĩa

* DVC tự động quyết định **khi nào cần train lại**
* Nếu data/code **không đổi** → không train
* Nếu đổi → sinh model mới

Chạy thủ công:

```bash
dvc repro
```

---

## 5. MLflow

MLflow dùng để:

* Log metric của từng mô hình
* So sánh 3 mô hình trong **1 run**
* Lưu thông tin model tốt nhất

### Chạy MLflow server

```bash
docker compose up mlflow
```

Truy cập:

```
http://localhost:5000
```

---

## 6. API Serving

* API viết bằng **FastAPI**
* Load model từ thư mục `models/`
* Không cần rebuild image khi có model mới

Chạy API:

```bash
docker compose up api
```

Test API:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product"}'
```

---

## 7. Docker Compose

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
```

---

## 8. Jenkins – CI/CD local-only

### Ý tưởng

* Jenkins **không clone Git remote**
* Jenkins **truy cập trực tiếp thư mục project local**
* Detect thay đổi bằng:

  * `git status`
  * `dvc status`

### Luồng Jenkins

```
Check local changes
   ├── Không đổi → STOP
   └── Có đổi → dvc repro → restart API
```

### Jenkinsfile (rút gọn)

```groovy
environment {
  PROJECT_DIR = "/absolute/path/to/sentiment_ml"
}
```

```bash
git status --porcelain
dvc status --quiet
dvc repro
docker compose restart api
```

---

## 9. Cách chạy nhanh (Quickstart)

```bash
# 1. Cài dependency
pip install -r requirements.txt

# 2. Init DVC (nếu chưa)
dvc init

# 3. Train model
dvc repro

# 4. Chạy hệ thống
docker compose up -d
```

---

## 10. Điểm mạnh của hệ thống

✔ Hoàn toàn local
✔ Không cần push Git / DVC
✔ Tự động train khi data đổi
✔ DVC đảm bảo reproducibility
✔ Jenkins đảm nhiệm CI/CD
✔ Phù hợp học MLOps & demo

---

## 11. Hướng phát triển

* Thêm DVC remote (S3 / GDrive)
* Thêm GitHub webhook
* Model versioning nâng cao
* Canary / A-B testing
* Chuyển sang Deep Learning

---

## 12. Tác giả

Nguyễn Tuấn Anh
MLOps / NLP / ML Systems

---

📌 *Dự án này được thiết kế để học và hiểu sâu bản chất MLOps, không phụ thuộc cloud.*
