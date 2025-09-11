FROM python:3.10-slim AS base

WORKDIR /app

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    unzip \
    && rm -rf /var/lib/apt/lists/*

    copy hybrid-adr-prediction.zip .
    RUN unzip hybrid-adr-prediction.zip -d ./&& rm hybrid-adr-prediction.zip
    WORKDIR /app/hybrid-adr-prediction-fresh
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt

    EXPOSE 8501
    CMD ["streamlit", "run", "app.py"]

