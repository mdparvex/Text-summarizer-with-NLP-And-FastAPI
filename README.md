# Summarizer Application (NLP + FastAPI + Celery + Redis + Docker)

This project is a text summarization service built with **FastAPI**, **Celery**, and **Redis**, containerized using **Docker Compose**.

## Project workflow description:
- collected dataset from [kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- preprocessed and clean data
- Trained on Attention based encoder decoder LSTM model
- Created an api to interact with the model
---

## 🚀 Features
- REST API with FastAPI
- Background task processing using Celery
- Redis as message broker & backend
- Example text summarization task
- Dockerized setup for easy deployment

---

## 📂 Project Structure
```
summarizer-app/
│── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── tasks.py         # Celery tasks
│   ├── worker.py        # Celery app instance
│   ├── summarizer.py    # Example summarizer logic
── models/
│   ├── decoder_models.h5
│   ├── encoder_model.h5
│   ├── reverse_target_word_index.pkl
│   ├── target_word_index.pkl
    ├── x_tokenizer.pkl
│── Dockerfile
│── Dockerfile.celery.summerizer_worker
│── docker-compose.yml
│── requirements.txt
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone 
cd summarizer-app
```


## ▶️ Running the Project

### Build & start containers
```bash
docker-compose up --build
```

### API Endpoints

#### 1. Submit text for summarization
```bash
POST http://localhost:8000/summarize/
{
  "text": "This is a long text that needs summarization."
}
```
Response:
```json
{
  "task_id": "12345",
  "status": "processing"
}
```

#### 2. Check result
```bash
GET http://localhost:8000/result/{task_id}
```
Response (when ready):
```json
{
  "task_id": "12345",
  "summary": "short summary"
}
```

---

