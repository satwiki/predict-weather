FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

# Default to Streamlit UI; override CMD for the API server
EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || curl --fail http://localhost:8000/health || exit 1

CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
