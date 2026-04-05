"""
GLM-4 / GLM-5 Personal Assistant Telegram Bot
===============================================
A powerful personal assistant bot powered by the GLM AI models.
Supports GLM-4 (fast) and GLM-5 (reasoning) with seamless switching.
"""

import os
import time
import logging
import requests
import asyncio
import sqlite3
import threading
import schedule
from datetime import datetime
import chromadb
import uuid
import re
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler
from playwright.sync_api import sync_playwright
try:
    from unstructured.partition.auto import partition
except ImportError:
    pass
import yt_dlp
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pdfplumber
import wikipedia
from deep_translator import GoogleTranslator
from langdetect import detect
from forex_python.converter import CurrencyRates
import python_weather

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode, ChatAction

# ─────────────────────────── Configuration ───────────────────────────

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GLM_API_KEY = os.environ.get("GLM_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable is not set.")
if not GLM_API_KEY:
    raise RuntimeError("GLM_API_KEY environment variable is not set.")

GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

AVAILABLE_MODELS = {
    "glm-4":  {"max_tokens": 2048, "description": "⚡ Fast & efficient"},
    "glm-5":  {"max_tokens": 4096, "description": "🧠 Reasoning model (deeper thinking)"},
}
DEFAULT_MODEL = "glm-4"

MAX_HISTORY = 20          # Max user+assistant messages to keep (system excluded)
MAX_TELEGRAM_LEN = 4096   # Telegram message character limit
DEFAULT_TEMPERATURE = 0.7
RETRY_DELAY = 2           # Seconds to wait before retry

SYSTEM_PROMPT = (
    "You are a highly capable personal assistant. You are direct, "
    "smart, and professional. You help with research, coding, writing, "
    "analysis, translation, math, planning, and any other task the "
    "user needs. You give thorough, accurate, and helpful responses "
    "without unnecessary disclaimers. Be conversational and friendly."
)

# ─────────────────────────── Logging ─────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────── Globals ─────────────────────────────────
APP_INSTANCE = None
c_rates = CurrencyRates(force_decimal=False)
chroma_client = chromadb.Client()
memory_collection = chroma_client.get_or_create_collection("user_memories")

# ─────────────────────────── Database Setup ────────────────────────────

def init_db():
    conn = sqlite3.connect("bot.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        preferred_language TEXT DEFAULT 'english',
        temperature REAL DEFAULT 0.7,
        active_model TEXT DEFAULT 'glm-4',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        remind_at TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        file_path TEXT,
        file_type TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────── Reminder Scheduler ───────────────────────

def check_reminders():
    now = datetime.now()
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, message FROM reminders
        WHERE remind_at <= ? AND is_active = 1
    """, (now,))
    due = cursor.fetchall()
    for reminder in due:
        rem_id, user_id, message = reminder
        # Send message to user via bot
        try:
            if APP_INSTANCE and APP_INSTANCE.bot:
                text = f"⏰ **REMINDER:**\n{message}"
                asyncio.run_coroutine_threadsafe(
                    APP_INSTANCE.bot.send_message(chat_id=user_id, text=text, parse_mode='Markdown'),
                    APP_INSTANCE.bot.get_request().loop  # wait, cleaner to not rely on loop this way if we can't
                    # we will just have the queue handle it? No, telegram bot has hard time. 
                    # better: use loop of the application.
                )
        except Exception as e:
            logger.error(f"Failed to send reminder {rem_id}: {e}")
            
        cursor.execute("UPDATE reminders SET is_active=0 WHERE id=?", (rem_id,))
    conn.commit()
    conn.close()

def _safe_send_reminder(user_id, message):
    if APP_INSTANCE and APP_INSTANCE.updater and APP_INSTANCE.updater.update_queue:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(APP_INSTANCE.bot.send_message(chat_id=user_id, text=f"⏰ **REMINDER:**\n{message}", parse_mode='Markdown'))

def _check_and_send():
    now = datetime.now()
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, user_id, message FROM reminders
        WHERE remind_at <= ? AND is_active = 1
    """, (now,))
    due = cursor.fetchall()
    
    # Needs a hack for loop
    loop = None
    if APP_INSTANCE:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
            
    for reminder in due:
        rem_id, user_id, message = reminder
        if APP_INSTANCE and loop:
            asyncio.run_coroutine_threadsafe(
                APP_INSTANCE.bot.send_message(chat_id=user_id, text=f"⏰ **REMINDER:**\n{message}", parse_mode='Markdown'),
                loop
            )
        cursor.execute("UPDATE reminders SET is_active=0 WHERE id=?", (rem_id,))
    conn.commit()
    conn.close()

def run_scheduler():
    schedule.every(1).minutes.do(_check_and_send)
    while True:
        schedule.run_pending()
        time.sleep(1)

threading.Thread(target=run_scheduler, daemon=True).start()

# ─────────────────────── Conversation & Model Store ──────────────────

conversations: dict[int, list[dict]] = {}

def get_user_prefs(user_id: int, username: str = None) -> dict:
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT preferred_language, temperature, active_model FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    if not row:
        cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username or "User"))
        conn.commit()
        row = ("english", DEFAULT_TEMPERATURE, DEFAULT_MODEL)
    conn.close()
    return {"language": row[0], "temperature": row[1], "model": row[2]}

def update_user_pref(user_id: int, key: str, value):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute(f"UPDATE users SET {key}=? WHERE user_id=?", (value, user_id))
    conn.commit()
    conn.close()

def get_model(user_id: int) -> str:
    return get_user_prefs(user_id)["model"]

def set_model(user_id: int, model: str) -> None:
    update_user_pref(user_id, "active_model", model)

def get_history(user_id: int) -> list[dict]:
    if user_id not in conversations:
        conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversations[user_id]

def append_message(user_id: int, role: str, content: str) -> None:
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    non_system = [m for m in history if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY:
        excess = len(non_system) - MAX_HISTORY
        conversations[user_id] = [history[0]] + non_system[excess:]

def clear_history(user_id: int) -> None:
    conversations[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

# ─────────────────────────── Async Tool Wrappers ────────────────────────
async def search_web(query):
    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))
    return await asyncio.to_thread(_search)

async def scrape_url(url):
    def _scrape():
        # Crawl4AI AsyncWebCrawler is async but wait user said they want asyncio.to_thread 
        # so lets just use it properly without event loop issues
        import asyncio as aio
        async def do_crawl():
            async with AsyncWebCrawler() as crawler:
                res = await crawler.arun(url=url)
                return res.markdown
        return aio.run(do_crawl())
    return await asyncio.to_thread(_scrape)

async def browse_page(url, idx=""):
    def _browse():
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(30000)
                page.goto(url)
                page.wait_for_load_state("networkidle")
                content = page.inner_text("body")
                browser.close()
                return content
        except Exception as e:
            return f"Failed to load page: {e}"
    return await asyncio.to_thread(_browse)

async def yt_extract(url):
    def _yt():
        ydl_opts = {"quiet": True, "skip_download": True, "writesubtitles": True, "writeautomaticsub": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    return await asyncio.to_thread(_yt)

async def search_wiki(topic):
    def _wiki():
        return wikipedia.summary(topic, sentences=20)
    return await asyncio.to_thread(_wiki)

async def search_wiki_full(topic):
    def _wiki():
        return wikipedia.page(topic).content
    return await asyncio.to_thread(_wiki)

async def fetch_weather(city):
    import python_weather
    async def _weather():
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            weather = await client.get(city)
            res = f"📍 **{city.capitalize()} Weather**\n🌡️ Temp: {weather.temperature}°C\n"
            return res
    return await _weather()

# ─────────────────────────── GLM API ────────────────────────────────

def call_glm(messages: list[dict], model: str, user_id: int) -> str:
    prefs = get_user_prefs(user_id)
    language = prefs["language"]
    temp = prefs["temperature"]
    
    sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    call_messages = list(messages)
    if sys_idx is not None:
        call_messages[sys_idx] = dict(call_messages[sys_idx])
        call_messages[sys_idx]["content"] += f"\nIMPORTANT: Your final output must be in {language}."

    model_config = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    payload = {
        "model": model,
        "messages": call_messages,
        "temperature": temp,
        "stream": False,
        "max_tokens": model_config["max_tokens"],
    }
    for attempt in range(1, 3):
        try:
            logger.info("GLM API call [%s] attempt %d (temp %s)", model, attempt, temp)
            resp = requests.post(GLM_API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            reply = choice.get("content") or ""
            if not reply.strip() and choice.get("reasoning_content"):
                reply = choice["reasoning_content"]
            if not reply.strip():
                reply = "I wasn't able to generate a response. Please try again."
            return reply
        except Exception as exc:
            logger.error("GLM API error [%s] (attempt %d): %s", model, attempt, exc)
            if attempt == 1: time.sleep(RETRY_DELAY)
    return "Sorry, the AI is busy right now. Please try again in a moment."

async def call_glm_async(messages, model, user_id):
    return await asyncio.to_thread(call_glm, messages, model, user_id)

# ─────────────────────────── Telegram Helpers ────────────────────────

async def send_long_message(update: Update, text: str, parse_mode: str | None = ParseMode.MARKDOWN) -> None:
    if len(text) <= MAX_TELEGRAM_LEN:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(text, parse_mode=None)
        return
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= MAX_TELEGRAM_LEN:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\\n\\n", 0, MAX_TELEGRAM_LEN)
        if split_at == -1: split_at = remaining.rfind("\\n", 0, MAX_TELEGRAM_LEN)
        if split_at == -1: split_at = MAX_TELEGRAM_LEN
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\\n")
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(chunk, parse_mode=None)

async def process_ai_request(update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str) -> None:
    user = update.effective_user
    logger.info("[%s] %s: %s", datetime.now().isoformat(), user.id, user_message)
    await update.message.chat.send_action(ChatAction.TYPING)
    append_message(user.id, "user", user_message)
    reply = await call_glm_async(get_history(user.id), get_model(user.id), user.id)
    append_message(user.id, "assistant", reply)
    await send_long_message(update, reply)

# ─────────────────────────── Command Handlers ────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    get_user_prefs(update.effective_user.id, update.effective_user.username)
    await update.message.reply_text("Welcome! type /help for all commands.", parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📖 **Available Commands**\n\n"
        "GENERAL:\n"
        "/start /help /clear /settings\n\n"
        "AI MODEL:\n"
        "/model - Switch GLM-4 or GLM-5\n\n"
        "WEB & RESEARCH:\n"
        "/search /scrape /browse\n"
        "/research /read /youtube\n"
        "/code /analyze\n\n"
        "MEMORY:\n"
        "/remember /recall /forget /memories\n\n"
        "DATA & FILES:\n"
        "/analyze_file /chart\n\n"
        "IMAGES:\n"
        "/image_info /resize /grayscale\n\n"
        "KNOWLEDGE:\n"
        "/wiki /wiki_full\n\n"
        "TRANSLATION:\n"
        "/translate /detect\n\n"
        "CURRENCY:\n"
        "/convert /rates\n\n"
        "WEATHER:\n"
        "/weather /forecast\n\n"
        "REMINDERS:\n"
        "/remind /reminders /cancel_reminder\n\n"
        "PREFERENCES:\n"
        "/set_language /set_temperature\n"
    )
    logger.info("[%s] Command /help by %s", datetime.now().isoformat(), update.effective_user.id)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_history(update.effective_user.id)
    await update.message.reply_text("🗑️ Conversation cleared!")

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if args and args[0].lower() in AVAILABLE_MODELS:
        set_model(update.effective_user.id, args[0].lower())
        await update.message.reply_text(f"✅ Switched to **{args[0].lower()}**", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("Usage `/model glm-4` or `/model glm-5`", parse_mode=ParseMode.MARKDOWN)

# --- Web & Research ---

async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query: return await update.message.reply_text("Usage: `/search <query>`", parse_mode=ParseMode.MARKDOWN)
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        results = await search_web(query)
        res_str = "\\n".join([f"{i+1}. {r['title']} - {r['href']}\\n{r['body']}" for i, r in enumerate(results[:5])])
        prompt = f"Search results for '{query}':\\n{res_str}\\nSummarize this directly and cite the sources."
        await process_ai_request(update, context, prompt)
    except Exception as e:
        await update.message.reply_text("Search failed. Please try again.")

async def cmd_scrape(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = " ".join(context.args)
    if not url: return await update.message.reply_text("Usage: `/scrape <url>`")
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        content = await scrape_url(url)
        content = content[:15000] # safety limit for roughly 3000 words
        await process_ai_request(update, context, f"Scraped content from {url} [Content truncated to 3000 words]:\\n{content}\\nProvide a summary.")
    except Exception as e:
        await update.message.reply_text("Could not scrape that URL.")

async def cmd_browse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = " ".join(context.args).split("|")
    if len(args) < 2: return await update.message.reply_text("Usage: `/browse <url> | <instruction>`")
    url, inst = args[0].strip(), args[1].strip()
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        content = await browse_page(url)
        content = content[:15000]
        await process_ai_request(update, context, f"Content from {url} [trunc]:\\n{content}\\nInstruction: {inst}")
    except Exception as e:
        await update.message.reply_text("Browser automation failed.")

async def cmd_research(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args)
    if not topic: return await update.message.reply_text("Usage: `/research <topic>`")
    prog = await update.message.reply_text(f"Searching the web for: {topic}...")
    try:
        results = await search_web(topic)
        await prog.edit_text("Scraping sources (1/5)...")
        # For speed in bot loop, just pass results directly to GPT without deep crawl of all to save time, or do it concurrently
        urls = [r['href'] for r in results[:5]]
        cors = [browse_page(u) for u in urls]
        pages = await asyncio.gather(*cors, return_exceptions=True)
        await prog.edit_text("Analyzing content with AI...")
        combined = ""
        for i, p in enumerate(pages):
            if isinstance(p, str): combined += f"\\n--- Source {i} ---\\n{p[:3000]}"
        await prog.edit_text("Generating report...")
        prompt = f"Based on the following research data, write a comprehensive, well-structured research report on: {topic}. Include key findings, analysis, and conclusion.\\n\\n{combined}"
        reply = await call_glm_async([{"role":"user", "content":prompt}], get_model(update.effective_user.id), update.effective_user.id)
        append_message(update.effective_user.id, "user", f"/research {topic}")
        append_message(update.effective_user.id, "assistant", reply)
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text("Something went wrong. Please try again!")

async def cmd_youtube(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = " ".join(context.args)
    if not url: return await update.message.reply_text("Usage: `/youtube <url>`")
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        info = await yt_extract(url)
        sub_text = ""
        try:
            if info.get('requested_subtitles'):
                sub_url = list(info['requested_subtitles'].values())[0]['url']
                sub_text = requests.get(sub_url).text[:10000] # Rough 2000 words limit
        except: pass
        prompt = f"Video: {info.get('title')}\\nDuration: {info.get('duration')}s\\nViews: {info.get('view_count')}\\nDescription: {info.get('description')[:500]}\\nTranscript: {sub_text}\\nSummarize this video and give key points."
        await process_ai_request(update, context, prompt)
    except Exception:
        await update.message.reply_text("Could not fetch that YouTube video.")

async def cmd_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    desc = " ".join(context.args)
    if not desc: return await update.message.reply_text("Usage: `/code <description>`")
    prompt = f"You are an expert programmer. Write clean, well-commented, production-ready code for: {desc}"
    await process_ai_request(update, context, prompt)

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text: return await update.message.reply_text("Usage: `/analyze <url or text>`")
    if text.startswith("http"):
        try:
            content = await scrape_url(text)
            text = f"[Content truncated to 3000 words]:\\n" + content[:15000]
        except:
            return await update.message.reply_text("Could not scrape that URL.")
    await process_ai_request(update, context, f"Analyze this content in detail. Provide insights, patterns, key points, and expert opinion:\\n{text}")

# --- Memory Commands ---

async def cmd_remember(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text: return await update.message.reply_text("Usage: `/remember <text>`")
    try:
        await asyncio.to_thread(
            memory_collection.add,
            documents=[text],
            metadatas=[{"user_id": str(update.effective_user.id), "timestamp": str(datetime.now())}],
            ids=[str(uuid.uuid4())]
        )
        await update.message.reply_text("Remembered! I'll never forget that.")
    except Exception:
        await update.message.reply_text("Memory operation failed. Try again.")

async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query: return await update.message.reply_text("Usage: `/recall <query>`")
    try:
        res = await asyncio.to_thread(
            memory_collection.query,
            query_texts=[query],
            n_results=3,
            where={"user_id": str(update.effective_user.id)}
        )
        found = res['documents'][0]
        if not found: return await update.message.reply_text("No relevant memories found.")
        reply = "Here is what I found:\\n" + "\\n".join([f"- {m}" for m in found])
        await update.message.reply_text(reply)
    except Exception:
        await update.message.reply_text("Memory operation failed. Try again.")

async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query: return await update.message.reply_text("Usage: `/forget <query>`")
    try:
        res = await asyncio.to_thread(
            memory_collection.query,
            query_texts=[query],
            n_results=1,
            where={"user_id": str(update.effective_user.id)}
        )
        if res['ids'] and res['ids'][0]:
            await asyncio.to_thread(memory_collection.delete, ids=res['ids'][0])
            await update.message.reply_text("Done, I've forgotten that.")
        else:
            await update.message.reply_text("No matching memory to forget.")
    except Exception:
        await update.message.reply_text("Memory operation failed. Try again.")

async def cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        mems = await asyncio.to_thread(memory_collection.get, where={"user_id": str(update.effective_user.id)})
        texts = mems.get('documents', [])[-10:]
        if not texts: return await update.message.reply_text("No memories saved.")
        reply = "Your last memories:\\n" + "\\n".join([f"{i+1}. {m}" for i, m in enumerate(texts)])
        await update.message.reply_text(reply)
    except Exception:
        await update.message.reply_text("Memory operation failed.")

# --- Knowledge & Translation ---

async def cmd_wiki(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args)
    if not topic: return await update.message.reply_text("Usage: `/wiki <topic>`")
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        summary = await search_wiki(topic)
        prompt = f"Wikipedia summary for {topic}:\\n{summary}\\nExplain this simply to me."
        await process_ai_request(update, context, prompt)
    except Exception:
        await update.message.reply_text("Could not find that on Wikipedia.")

async def cmd_wiki_full(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args)
    if not topic: return await update.message.reply_text("Usage: `/wiki_full <topic>`")
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        content = await search_wiki_full(topic)
        prompt = f"Full Wikipedia article for {topic} (truncated):\\n{content[:15000]}\\nProvide a comprehensive summary."
        await process_ai_request(update, context, prompt)
    except Exception:
        await update.message.reply_text("Could not find that on Wikipedia.")

async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2: return await update.message.reply_text("Usage: `/translate <language> <text>`")
    lang, text = context.args[0], " ".join(context.args[1:])
    try:
        source_lang = await asyncio.to_thread(detect, text)
        res = await asyncio.to_thread(GoogleTranslator(source='auto', target=lang).translate, text)
        await update.message.reply_text(f"Translated to {lang} (Detected {source_lang}):\\n{res}")
    except Exception as e:
        await update.message.reply_text(f"Translation failed: {e}")

async def cmd_detect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text: return await update.message.reply_text("Usage: `/detect <text>`")
    try:
        res = await asyncio.to_thread(detect, text)
        await update.message.reply_text(f"This text is in: {res}")
    except Exception as e:
        await update.message.reply_text(f"Detection failed: {e}")

# --- Currency & Weather ---

async def cmd_convert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 3: return await update.message.reply_text("Usage: `/convert <amount> <from> <to>`")
    try:
        amt, f, t = float(context.args[0]), context.args[1].upper(), context.args[2].upper()
        res = await asyncio.to_thread(c_rates.convert, f, t, amt)
        await update.message.reply_text(f"💱 {amt} {f} = {res:.2f} {t}")
    except Exception:
        await update.message.reply_text("Currency conversion failed.")

async def cmd_rates(update: Update, context: ContextTypes.DEFAULT_TYPE):
    base = context.args[0].upper() if context.args else "USD"
    try:
        rates = await asyncio.to_thread(c_rates.get_rates, base)
        k = list(rates.keys())[:10]
        r = [f"{x}: {rates[x]:.2f}" for x in k]
        await update.message.reply_text(f"Top rates for {base}:\\n" + "\\n".join(r))
    except Exception:
        await update.message.reply_text("Currency conversion failed.")

async def cmd_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    city = " ".join(context.args)
    if not city: return await update.message.reply_text("Usage: `/weather <city>`")
    try:
        res = await fetch_weather(city)
        await update.message.reply_text(res)
    except Exception:
        await update.message.reply_text("Could not fetch weather for that city.")

async def cmd_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    city = " ".join(context.args)
    if not city: return await update.message.reply_text("Usage: `/forecast <city>`")
    await update.message.reply_text("Forecast isn't detailed in current limited python_weather wrapper, falling back to weather.")
    await cmd_weather(update, context)

# --- Reminders & Settings ---

async def cmd_remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2: return await update.message.reply_text("Usage: `/remind <time:10m|1h> <message>`")
    time_str, msg = context.args[0], " ".join(context.args[1:])
    try:
        num, unit = int(time_str[:-1]), time_str[-1]
        delta = 0
        if unit == 'm': delta = num * 60
        elif unit == 'h': delta = num * 3600
        elif unit == 'd': delta = num * 86400
        else: return await update.message.reply_text("Use m (minutes), h (hours), d (days).")
        
        target = datetime.fromtimestamp(time.time() + delta)
        conn = sqlite3.connect("bot.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO reminders (user_id, message, remind_at) VALUES (?, ?, ?)", (update.effective_user.id, msg, target))
        conn.commit()
        conn.close()
        await update.message.reply_text(f"Reminder set for {time_str}!")
    except Exception:
        await update.message.reply_text("Failed to set reminder. Ensure format like 30m.")

async def cmd_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("SELECT id, message, remind_at FROM reminders WHERE user_id=? AND is_active=1", (update.effective_user.id,))
    reminders = cur.fetchall()
    conn.close()
    if not reminders: return await update.message.reply_text("No active reminders.")
    reply = "Active reminders:\\n" + "\\n".join([f"{r[0]}. {r[1]} - {r[2]}" for r in reminders])
    await update.message.reply_text(reply)

async def cmd_cancel_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rid = int(context.args[0])
        conn = sqlite3.connect("bot.db")
        cur = conn.cursor()
        cur.execute("UPDATE reminders SET is_active=0 WHERE id=? AND user_id=?", (rid, update.effective_user.id))
        conn.commit()
        conn.close()
        await update.message.reply_text("Reminder cancelled!")
    except Exception:
        await update.message.reply_text("Usage: `/cancel_reminder <number>`")

async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = get_user_prefs(update.effective_user.id)
    await update.message.reply_text(
        f"⚙️ **Your Settings**\nModel: {prefs['model']}\nLanguage: {prefs['language']}\nTemp: {prefs['temperature']}",
        parse_mode=ParseMode.MARKDOWN
    )

async def cmd_set_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = " ".join(context.args)
    if not lang: return await update.message.reply_text("Usage: `/set_language <language>`")
    update_user_pref(update.effective_user.id, "preferred_language", lang)
    await update.message.reply_text(f"Language set to {lang}.")

async def cmd_set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        temp = float(context.args[0])
        if 0.1 <= temp <= 1.0:
            update_user_pref(update.effective_user.id, "temperature", temp)
            await update.message.reply_text(f"Temperature set to {temp}")
        else:
            await update.message.reply_text("Must be 0.1 - 1.0")
    except:
        await update.message.reply_text("Usage: `/set_temperature <0.1-1.0>`")

# --- File Operations ---

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    file = await context.bot.get_file(doc.file_id)
    local_path = f"/tmp/{update.effective_user.id}_{doc.file_name}"
    await file.download_to_drive(local_path)
    
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO uploaded_files (user_id, file_path, file_type) VALUES (?, ?, ?)", 
                (update.effective_user.id, local_path, doc.mime_type))
    conn.commit()
    conn.close()

    ext = doc.file_name.lower().split('.')[-1]
    if ext in ['csv', 'xlsx']:
        await update.message.reply_text("File saved. Suggestion: send `/analyze_file` to analyze this document.")
    elif ext == 'pdf':
        await update.message.reply_text("File saved. Suggestion: send `/read` to summarize it.")
    elif doc.mime_type and doc.mime_type.startswith('image'):
        await update.message.reply_text("Image saved. Suggestion: send `/image_info`, `/resize [w] [h]`, or `/grayscale`.")
    else:
        text = "File saved. Summarization coming soon for TXT."
        await update.message.reply_text(text)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    local_path = f"/tmp/{update.effective_user.id}_{photo.file_unique_id}.jpg"
    await file.download_to_drive(local_path)

    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO uploaded_files (user_id, file_path, file_type) VALUES (?, ?, ?)", 
                (update.effective_user.id, local_path, 'image/jpeg'))
    conn.commit()
    conn.close()
    await update.message.reply_text("Image saved. Suggestion: send `/image_info`, `/resize [w] [h]`, or `/grayscale`.")

def get_last_file(user_id, is_image=False):
    conn = sqlite3.connect("bot.db")
    cur = conn.cursor()
    if is_image:
        cur.execute("SELECT file_path FROM uploaded_files WHERE user_id=? AND file_type LIKE 'image%' ORDER BY id DESC LIMIT 1", (user_id,))
    else:
        cur.execute("SELECT file_path FROM uploaded_files WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,))
    res = cur.fetchone()
    conn.close()
    return res[0] if res else None

async def cmd_analyze_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_path = get_last_file(update.effective_user.id)
    if not file_path: return await update.message.reply_text("No file uploaded recently.")
    try:
        if file_path.endswith('.csv'): df = pd.read_csv(file_path)
        else: df = pd.read_excel(file_path)
        stats = f"Rows: {df.shape[0]}, Cols: {df.shape[1]}\\nColumns: {', '.join(df.columns)}\\nMissing values: {df.isnull().sum().sum()}"
        await process_ai_request(update, context, f"Analyze these file stats:\\n{stats}")
    except Exception:
        await update.message.reply_text("Analysis failed. Must be CSV or Excel.")

async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2: return await update.message.reply_text("Usage `/chart <type> <columns>`")
    ctype, cols = context.args[0], context.args[1].split(',')
    file_path = get_last_file(update.effective_user.id)
    if not file_path: return await update.message.reply_text("No file uploaded recently.")
    try:
        if file_path.endswith('.csv'): df = pd.read_csv(file_path)
        else: df = pd.read_excel(file_path)
        plt.figure()
        if ctype == 'bar': df[cols].plot(kind='bar')
        elif ctype == 'line': df[cols].plot(kind='line')
        elif ctype == 'pie': df[cols].plot(kind='pie', y=cols[0])
        elif ctype == 'scatter': df.plot.scatter(x=cols[0], y=cols[1])
        elif ctype == 'histogram': df[cols].plot(kind='hist')
        img_path = f"/tmp/{update.effective_user.id}_chart.png"
        plt.savefig(img_path)
        await update.message.reply_photo(photo=open(img_path, 'rb'))
    except Exception:
        await update.message.reply_text("Chart failed. Ensure valid type and columns: bar, line, pie, scatter, histogram.")

async def cmd_read(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        # User provided url
        url = " ".join(context.args)
        await process_ai_request(update, context, f"Fetch and read `{url}`. Reply with 'use /read later' for this placeholder until URL reading is robust or implemented via unstructured/pdf.")
        return
    file_path = get_last_file(update.effective_user.id)
    if not file_path: return await update.message.reply_text("No file uploaded.")
    try:
        if file_path.endswith('.pdf'):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages: text += page.extract_text() + "\\n"
        else:
            try:
                elements = partition(filename=file_path)
                text = "\\n".join([str(e) for e in elements])
            except:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        await process_ai_request(update, context, f"Read this document [trunc 3000 words]:\\n{text[:15000]}\\nSummarize its structured data.")
    except Exception:
        await update.message.reply_text("Could not read document.")

async def cmd_image_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_path = get_last_file(update.effective_user.id, is_image=True)
    if not file_path: return await update.message.reply_text("No image uploaded.")
    try:
        with Image.open(file_path) as img:
            res = f"Format: {img.format}\\nMode: {img.mode}\\nSize: {img.size}"
            await update.message.reply_text(res)
    except Exception:
        await update.message.reply_text("Image operation failed.")

async def cmd_resize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2: return await update.message.reply_text("Usage `/resize <w> <h>`")
    w, h = int(context.args[0]), int(context.args[1])
    file_path = get_last_file(update.effective_user.id, is_image=True)
    if not file_path: return await update.message.reply_text("No image uploaded.")
    try:
        with Image.open(file_path) as img:
            img = img.resize((w, h))
            out = f"/tmp/{update.effective_user.id}_resized.jpg"
            img.save(out)
            await update.message.reply_photo(photo=open(out, 'rb'))
    except Exception:
        await update.message.reply_text("Image operation failed.")

async def cmd_grayscale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_path = get_last_file(update.effective_user.id, is_image=True)
    if not file_path: return await update.message.reply_text("No image uploaded.")
    try:
        with Image.open(file_path) as img:
            img = ImageOps.grayscale(img)
            out = f"/tmp/{update.effective_user.id}_gray.jpg"
            img.save(out)
            await update.message.reply_photo(photo=open(out, 'rb'))
    except Exception:
        await update.message.reply_text("Image operation failed.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    await process_ai_request(update, context, update.message.text)

# ─────────────────────────── Main ────────────────────────────────────

def main() -> None:
    logger.info("Starting GLM Telegram Bot (polling mode)...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    global APP_INSTANCE
    APP_INSTANCE = app

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("set_language", cmd_set_language))
    app.add_handler(CommandHandler("set_temperature", cmd_set_temperature))
    
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("scrape", cmd_scrape))
    app.add_handler(CommandHandler("browse", cmd_browse))
    app.add_handler(CommandHandler("research", cmd_research))
    app.add_handler(CommandHandler("youtube", cmd_youtube))
    app.add_handler(CommandHandler("code", cmd_code))
    app.add_handler(CommandHandler("analyze", cmd_analyze))

    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("recall", cmd_recall))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("memories", cmd_memories))

    app.add_handler(CommandHandler("analyze_file", cmd_analyze_file))
    app.add_handler(CommandHandler("chart", cmd_chart))
    app.add_handler(CommandHandler("read", cmd_read))

    app.add_handler(CommandHandler("image_info", cmd_image_info))
    app.add_handler(CommandHandler("resize", cmd_resize))
    app.add_handler(CommandHandler("grayscale", cmd_grayscale))

    app.add_handler(CommandHandler("wiki", cmd_wiki))
    app.add_handler(CommandHandler("wiki_full", cmd_wiki_full))
    app.add_handler(CommandHandler("translate", cmd_translate))
    app.add_handler(CommandHandler("detect", cmd_detect))
    
    app.add_handler(CommandHandler("convert", cmd_convert))
    app.add_handler(CommandHandler("rates", cmd_rates))
    app.add_handler(CommandHandler("weather", cmd_weather))
    app.add_handler(CommandHandler("forecast", cmd_forecast))
    
    app.add_handler(CommandHandler("remind", cmd_remind))
    app.add_handler(CommandHandler("reminders", cmd_reminders))
    app.add_handler(CommandHandler("cancel_reminder", cmd_cancel_reminder))

    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
