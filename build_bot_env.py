import os

def update_reqs():
    reqs = """python-telegram-bot==20.7
requests
httpx
crawl4ai
playwright
duckduckgo-search
unstructured
yt-dlp
chromadb
pandas
matplotlib
Pillow
pdfplumber
openpyxl
wikipedia
deep-translator
langdetect
schedule
forex-python
python-weather
RestrictedPython
reportlab
fpdf2
python-docx
pytesseract
openai-whisper
sqlalchemy
fastapi
uvicorn
python-multipart
numpy<2
"""
    with open("requirements.txt", "w") as f:
        f.write(reqs)

def update_docker():
    df = """FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \\
    wget curl ffmpeg build-essential gcc g++ python3-dev \\
    libglib2.0-0 libnss3 libnspr4 \\
    libatk1.0-0 libatk-bridge2.0-0 \\
    libcups2 libdrm2 libdbus-1-3 \\
    libxcb1 libxkbcommon0 libx11-6 \\
    libxcomposite1 libxdamage1 libxext6 \\
    libxfixes3 libxrandr2 libgbm1 \\
    libpango-1.0-0 libcairo2 libasound2 \\
    libfreetype6 libfontconfig1 \\
    tesseract-ocr tesseract-ocr-all \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium
RUN crawl4ai-setup

COPY bot.py .

EXPOSE 8080

CMD ["python", "-u", "bot.py"]
"""
    with open("Dockerfile", "w", newline='\n') as f:
        f.write(df)

if __name__ == "__main__":
    update_reqs()
    update_docker()
