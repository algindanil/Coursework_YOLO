import telebot

import os
from pathlib import Path
from dotenv import load_dotenv

from run_inference import run_yolo

load_dotenv()

bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))

available_weights = {
    'common': os.getenv('COMMON_WEIGHTS_PATH'),
    'aircraft': os.getenv('AIRCRAFT_WEIGHTS_PATH')
}

active_weights_path = available_weights['aircraft']


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Hi! Send a photo or a video of a military aircraft and I will try to detect it!')


@bot.message_handler(commands=['setmodel'])
def set_model(message):
    global active_weights_path
    try:
        model_name = message.text.split()[1]
        if model_name in available_weights:
            active_weights_path = available_weights[model_name]
            bot.reply_to(message, f"Model set to {model_name}.")
        else:
            bot.reply_to(message, "Model not found. Available models are: " + ", ".join(available_weights.keys()))
    except IndexError:
        bot.reply_to(message, "Usage: /setmodel <modelname>")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    photo_path = Path(f'input_images/{file_info.file_id}.jpg')
    photo_path.parent.mkdir(parents=True, exist_ok=True)

    with open(photo_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    result_image_path = run_yolo(active_weights_path, photo_path, file_info.file_id)

    with open(result_image_path / Path(f'{file_info.file_id}.jpg'), 'rb') as result_image:
        bot.send_photo(message.chat.id, result_image)


bot.polling()
