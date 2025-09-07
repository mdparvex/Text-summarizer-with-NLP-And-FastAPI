from app.summarizer import decode_sequence
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from app.worker import celery_app

base_dir = Path(__file__).parent.parent
with open(base_dir.joinpath("models","x_tokenizer.pkl"), "rb") as f:
    x_tokenizer = pickle.load(f)

max_len_text = 80      # example
@celery_app.task
def summarize_task(text: str):
    # Convert input text → sequence
    print(f'text: {text}')
    seq = x_tokenizer.texts_to_sequences([text])
    print(f'seq: {seq}')
    seq = pad_sequences(seq, maxlen=max_len_text, padding='post')
    # Decode summary
    summary = decode_sequence(seq.reshape(1,max_len_text))
    print(f'summery: {summary}')
    return summary
