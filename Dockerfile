FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002
CMD ["python", "app.py"]