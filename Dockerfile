FROM python:3.9-slim

# Set memory optimization environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV PYTHONGC=1

WORKDIR /app

# Install system dependencies with cleanup
RUN apt-get update && apt-get install -y \
    ffmpeg \
    procps \
    htop \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Add psutil for monitoring
RUN pip install --no-cache-dir psutil

# Copy application files
COPY . .

# Create healthcheck script
RUN echo '#!/usr/bin/env python3\nimport sys, requests, os, psutil\ntry:\n  r = requests.get("http://localhost:5002/refertazione/heartbeat", timeout=10)\n  if r.status_code == 200:\n    mem = psutil.virtual_memory()\n    if mem.percent > 95:\n      print("Memory too high:", mem.percent)\n      sys.exit(1)\n    print("Healthy")\n    sys.exit(0)\n  else:\n    sys.exit(1)\nexcept:\n  sys.exit(1)' > /app/healthcheck.py && chmod +x /app/healthcheck.py

# Health check that tests actual functionality
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=3 \
  CMD python /app/healthcheck.py

EXPOSE 5002

# Use proper signal handling
CMD ["python", "-u", "app.py"]