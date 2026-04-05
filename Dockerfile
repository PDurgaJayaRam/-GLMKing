FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget curl ffmpeg build-essential gcc g++ python3-dev \
    libglib2.0-0 libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 \
    libxcb1 libxkbcommon0 libx11-6 \
    libxcomposite1 libxdamage1 libxext6 \
    libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 \
    libfreetype6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium
RUN crawl4ai-setup

COPY bot.py .
CMD ["python", "-u", "bot.py"]
