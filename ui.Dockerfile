FROM python:3.11-slim

WORKDIR /ui

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt /ui/requirements.txt
RUN pip install --no-cache-dir -r /ui/requirements.txt

COPY ui /ui

EXPOSE 7860

CMD ["python", "gradio_app.py"]
