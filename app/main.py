# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.tasks import summarize_task
from app.worker import celery_app


app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/summarize/")
def summarize(req: TextRequest):
    task = summarize_task.delay(req.text)
    return {"task_id": task.id, "status": "processing"}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if task_result.failed():
        return {"task_id": task_id, "status": "failed", "error": str(task_result.result)}
    if task_result.ready():
        return {"task_id": task_id, "summary": task_result.result}
    return {"task_id": task_id, "status": "processing"}
# from pathlib import Path
# import pickle
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from app.summarizer import decode_sequence
# base_dir = Path(__file__).parent.parent
# with open(base_dir.joinpath("models","x_tokenizer.pkl"), "rb") as f:
#     x_tokenizer = pickle.load(f)

# max_len_text = 80      # example
# @app.post("/summarize/old/")
# async def summarize(req: TextRequest):
#     # Convert input text → sequence
#     seq = x_tokenizer.texts_to_sequences([req.text])
#     seq = pad_sequences(seq, maxlen=max_len_text, padding='post')

#     # Decode summary
#     summary = decode_sequence(seq.reshape(1,max_len_text))

#     return {"summary": summary}

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

