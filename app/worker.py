import os
from celery import Celery

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "summarizer",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["app.tasks"],
)

celery_app.conf.task_track_started = True
