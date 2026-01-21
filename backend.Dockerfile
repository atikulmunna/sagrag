FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc wget curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip + configure retry limits
RUN pip install --upgrade pip
RUN pip config set global.index-url https://pypi.org/simple
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cu121
RUN pip config set global.timeout 2000
RUN pip config set global.retries 20

# Copy requirements
COPY app/requirements.txt /app/requirements.txt

# Install requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install spaCy model
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Copy app code
COPY app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
