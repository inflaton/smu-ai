import os
import ssl
import time
from threading import Thread

import requests
from telegram import Update
from telegram import __version__ as TG_VER
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app_modules.init import app_init

llm_loader, qa_chain = app_init()

ctx = ssl.create_default_context()
ctx.set_ciphers("DEFAULT")

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

TOKEN = os.getenv("TELEGRAM_API_TOKEN")


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! You are welcome to ask questions on anything!",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    print(update)
    tic = time.perf_counter()
    try:
        result = qa_chain.call_chain(
            {"question": update.message.text, "chat_history": []}, None
        )

        result = result["answer"]
        print(result)
        await update.message.reply_text(result[0:8192])
        toc = time.perf_counter()
        print(f"Response time in {toc - tic:0.4f} seconds")
    except Exception as e:
        print("error", e)


def start_telegram_bot() -> None:
    """Start the bot."""
    print("starting telegram bot ...")
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start_command", start_command))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - chat_command the message on Telegram
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat_command)
    )

    application.run_polling()


if __name__ == "__main__":
    start_telegram_bot()
