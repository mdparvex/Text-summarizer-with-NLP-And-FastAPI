FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ./app ./app
COPY ./models ./models

# Expose FastAPI port
EXPOSE 8000

# Default command (overridden in docker-compose for workers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
