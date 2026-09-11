FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV BOOK_STORE=/data BOT_MODE=paper PYTHONPATH=/app
VOLUME ["/data"]
EXPOSE 8000
# Bound to 0.0.0.0 only because the container's port is what gets published.
# Publish it to 127.0.0.1 on the host, or put an authenticating proxy in front:
#   docker run -p 127.0.0.1:8000:8000 ...
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
