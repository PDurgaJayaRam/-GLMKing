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
DEFAULT_MODEL = "glm-5"

MAX_HISTORY = 20          # Max user+assistant messages to keep (system excluded)
MAX_TELEGRAM_LEN = 4096   # Telegram message character limit
TEMPERATURE = 0.7
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

# ─────────────────────── Conversation & Model Store ──────────────────

conversations: dict[int, list[dict]] = {}
user_models: dict[int, str] = {}


def get_model(user_id: int) -> str:
    """Return the active model for a user (default: GLM-5)."""
    return user_models.get(user_id, DEFAULT_MODEL)


def set_model(user_id: int, model: str) -> None:
    """Set the active model for a user."""
    user_models[user_id] = model


def get_history(user_id: int) -> list[dict]:
    """Return (and lazily initialize) conversation history for a user."""
    if user_id not in conversations:
        conversations[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return conversations[user_id]


def append_message(user_id: int, role: str, content: str) -> None:
    """Append a message and trim history to MAX_HISTORY (excluding system)."""
    history = get_history(user_id)
    history.append({"role": role, "content": content})

    # Keep system prompt + last MAX_HISTORY messages
    non_system = [m for m in history if m["role"] != "system"]
    if len(non_system) > MAX_HISTORY:
        excess = len(non_system) - MAX_HISTORY
        conversations[user_id] = [history[0]] + non_system[excess:]


def clear_history(user_id: int) -> None:
    """Reset a user's conversation history."""
    conversations[user_id] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]


# ─────────────────────────── GLM API ────────────────────────────────

def call_glm(messages: list[dict], model: str) -> str:
    """
    Call the GLM chat completions API.
    Handles both GLM-4 (standard) and GLM-5 (reasoning) response formats.
    Retries once on failure after RETRY_DELAY seconds.
    """
    model_config = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GLM_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": False,
        "max_tokens": model_config["max_tokens"],
    }

    for attempt in range(1, 3):  # attempt 1 and 2
        try:
            logger.info("GLM API call [%s] attempt %d", model, attempt)
            resp = requests.post(
                GLM_API_URL, json=payload, headers=headers, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]["message"]

            # GLM-5 returns reasoning_content + content
            # GLM-4 returns just content
            reply = choice.get("content") or ""

            # If content is empty but reasoning exists (edge case: truncated),
            # fall back to reasoning content
            if not reply.strip() and choice.get("reasoning_content"):
                reply = choice["reasoning_content"]

            if not reply.strip():
                reply = "I wasn't able to generate a response. Please try again."

            logger.info("GLM API success [%s] (attempt %d)", model, attempt)
            return reply

        except Exception as exc:
            logger.error("GLM API error [%s] (attempt %d): %s", model, attempt, exc)
            if attempt == 1:
                time.sleep(RETRY_DELAY)

    return "Sorry, the AI is busy right now. Please try again in a moment."


# ─────────────────────────── Telegram Helpers ────────────────────────

async def send_long_message(
    update: Update, text: str, parse_mode: str | None = ParseMode.MARKDOWN
) -> None:
    """Split and send messages that exceed Telegram's character limit."""
    if len(text) <= MAX_TELEGRAM_LEN:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
        except Exception:
            # Fallback: send without Markdown if parsing fails
            await update.message.reply_text(text, parse_mode=None)
        return

    # Split on double-newlines first, then single newlines, then hard-split
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= MAX_TELEGRAM_LEN:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, MAX_TELEGRAM_LEN)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, MAX_TELEGRAM_LEN)
        if split_at == -1:
            split_at = MAX_TELEGRAM_LEN
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(chunk, parse_mode=None)


async def process_ai_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_message: str
) -> None:
    """Common flow: typing indicator → GLM call → reply."""
    user = update.effective_user
    user_id = user.id
    model = get_model(user_id)
    logger.info("[USER %s] [%s] %s", user_id, model, user_message)

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    # Add user message and call API
    append_message(user_id, "user", user_message)
    reply = call_glm(get_history(user_id), model)
    append_message(user_id, "assistant", reply)

    logger.info("[BOT  %s] [%s] %s", user_id, model, reply[:120])

    await send_long_message(update, reply)


# ─────────────────────────── Command Handlers ────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — greet the user."""
    name = update.effective_user.first_name or "there"
    model = get_model(update.effective_user.id)
    text = (
        f"Hey {name}! 👋 I'm your personal AI assistant powered by **{model.upper()}**.\n\n"
        "I can help you with:\n"
        "• 🔍 Research & summarization\n"
        "• 💻 Coding & debugging\n"
        "• ✍️ Writing & editing\n"
        "• 🌐 Translation (any language)\n"
        "• 📊 Math & analysis\n"
        "• 📋 Planning & brainstorming\n\n"
        "Just send me a message to get started!\n"
        "Use /model to switch between GLM-4 and GLM-5."
    )
    logger.info("[CMD  %s] /start", update.effective_user.id)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — list available commands."""
    model = get_model(update.effective_user.id)
    text = (
        "📖 **Available Commands**\n\n"
        "`/start` — Welcome message\n"
        "`/help` — Show this help menu\n"
        "`/clear` — Clear conversation history\n"
        "`/model` — Switch between GLM-4 and GLM-5\n"
        "`/research <topic>` — Deep research report on a topic\n"
        "`/code <description>` — Generate clean, commented code\n"
        "`/translate <lang> <text>` — Translate text to any language\n"
        "`/summarize <text>` — Summarize provided text\n"
        "`/imagine <description>` — Creative writing piece\n\n"
        f"🤖 Active model: **{model}**\n"
        "Or just send any message to chat freely!"
    )
    logger.info("[CMD  %s] /help", update.effective_user.id)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear — reset conversation history."""
    clear_history(update.effective_user.id)
    logger.info("[CMD  %s] /clear", update.effective_user.id)
    await update.message.reply_text("🗑️ Conversation cleared! Starting fresh.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model — switch between GLM-4 and GLM-5."""
    user_id = update.effective_user.id
    args = context.args

    if args and args[0].lower() in AVAILABLE_MODELS:
        chosen = args[0].lower()
        set_model(user_id, chosen)
        info = AVAILABLE_MODELS[chosen]
        await update.message.reply_text(
            f"✅ Switched to **{chosen}**\n{info['description']}",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("[CMD  %s] /model → %s", user_id, chosen)
        return

    # Show model picker
    current = get_model(user_id)
    lines = ["🤖 **Select a Model**\n"]
    for name, info in AVAILABLE_MODELS.items():
        marker = " ← current" if name == current else ""
        lines.append(f"• `{name}` — {info['description']}{marker}")
    lines.append(f"\nUsage: `/model glm-4` or `/model glm-5`")
    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.MARKDOWN
    )


async def cmd_research(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /research <topic>."""
    topic = " ".join(context.args) if context.args else ""
    if not topic:
        await update.message.reply_text(
            "Usage: `/research <topic>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    await process_ai_request(
        update, context, f"Do a thorough research report on: {topic}"
    )


async def cmd_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /code <description>."""
    desc = " ".join(context.args) if context.args else ""
    if not desc:
        await update.message.reply_text(
            "Usage: `/code <description>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    await process_ai_request(
        update, context, f"Write clean, well-commented code for: {desc}"
    )


async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /translate <language> <text>."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: `/translate <language> <text>`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    language = context.args[0]
    text = " ".join(context.args[1:])
    await process_ai_request(
        update, context, f"Translate the following text to {language}: {text}"
    )


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summarize <text>."""
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "Usage: `/summarize <text>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    await process_ai_request(
        update, context, f"Summarize the following: {text}"
    )


async def cmd_imagine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /imagine <description>."""
    desc = " ".join(context.args) if context.args else ""
    if not desc:
        await update.message.reply_text(
            "Usage: `/imagine <description>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    await process_ai_request(
        update,
        context,
        f"Generate a detailed, vivid, and creative writing piece based on this description: {desc}",
    )


# ─────────────────────────── Message Handler ─────────────────────────

async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle plain text messages."""
    if not update.message or not update.message.text:
        return
    await process_ai_request(update, context, update.message.text)


# ─────────────────────────── Main ────────────────────────────────────

def main() -> None:
    logger.info("Starting GLM Telegram Bot (polling mode)...")
    logger.info("Default model: %s", DEFAULT_MODEL)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("research", cmd_research))
    app.add_handler(CommandHandler("code", cmd_code))
    app.add_handler(CommandHandler("translate", cmd_translate))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(CommandHandler("imagine", cmd_imagine))

    # Register plain-text message handler
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Start polling (no webhook, no port needed)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
