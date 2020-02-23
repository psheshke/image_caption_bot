from model.torch_model import CaptionNet, BeheadedInception3
import torch, torch.nn as nn
import torch.nn.functional as F

import logging
from telegram import Bot, Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram_token import token, TG_API_URL, proxy
# from dota2_wiki_parser import parser
import torch
# from config import reply_texts
import numpy as np
from PIL import Image
from io import BytesIO
import time
# import urllib
import requests
# from model.beheaded_inception3 import beheaded_inception_v3
# inception = beheaded_inception_v3().train(False)


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = []
# open file and read the content in a list
with open('vocab.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        vocab.append(currentPlace)

start_symbol = '_START_'
end_symbol = '_END_'
unknown_symbol = '_UNK_'
padding_symbol = '_PAD_'

word_to_index = {w: i for i, w in enumerate(vocab)}
eos_ix = word_to_index[end_symbol]
unk_ix = word_to_index[unknown_symbol]
pad_ix = word_to_index[padding_symbol]


def word_to_matrix(sequences, max_len=None):
    max_len = max_len or max(map(len, sequences))

    matrix = np.ones((len(sequences), max_len), dtype='int32') * pad_ix
    for i, seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix

network = torch.load('../model/model100ep384bs.pkl')
network.eval()
inception = torch.load('../model/inception.pkl')
inception.eval()

def generate_caption(image, caption_prefix=("_START_",),
                     t=1, sample=True, max_len=100):
    assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
           and np.min(image) >= 0 and image.shape[-1] == 3

    with torch.no_grad():
        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
        vectors_8x8, vectors_neck, logits = inception(image[None])
        caption_prefix = list(caption_prefix)

        vectors_neck = vectors_neck.to(device)
        for _ in range(max_len):
            # представить в виде матрицы
            prefix_ix = word_to_matrix([caption_prefix])
            prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64).to(device)
            # получаем логиты из RNN-ки
            next_word_logits = network.forward(vectors_neck, prefix_ix)[0, -1]
            # переводим их в вероятности
            next_word_probs = F.softmax(next_word_logits, dim=-1).cpu().data.numpy()

            assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
            next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t)  # опционально

            if sample:
                next_word = np.random.choice(vocab, p=next_word_probs)
            else:
                next_word = vocab[np.argmax(next_word_probs)]

            caption_prefix.append(next_word)

            # RNN-ка сгенерила символ конца предложения, расходимся
            if next_word == "_END_":
                break

    return caption_prefix

def do_start(update, context):

    text = "Привет, я бот, который любит придумывать описания для картинок. \n " \
           "если ты пришлешь мне фото одного из них, \n " \
           "то я попрубую описать, что на нем изображено. \n " \
           "Попробуем?"

    update.message.reply_text(text)
#
def do_echo(update, context):

    start_time = time.process_time()

    text = update.message.text

    if ('http' in text and ('png' in text or 'jpg' in text)):

        update.message.reply_text(text='Хммм....дай подумать..', )

        response = requests.get([i for i in text.split(' ') if 'http' in i][0])

        img = BytesIO()

        Image.open(BytesIO(response.content)).convert('RGB').save(img, 'PNG')

        update.message.reply_text(text='Щас поглядимс..', )

        img = np.array(Image.fromarray(np.array(Image.open(img))).resize((299, 299))).astype('float32') / 255.

        text = 'Я бы сказал, что это: \n'
        for i in range(10):
            text += ' '.join(generate_caption(img, t=5.)[1:-1]) + '\n'

        update.message.reply_text(text=text, parse_mode='Markdown')

    else:

        update.message.reply_text("Если ты отправишь мне картинку файлом\n" \
                                  "или прямую ссылку на нее, то я попробую придумать описание к фото")

    end_time = time.process_time()
    print('Duration: {}'.format(end_time - start_time))

def send_prediction_on_photo(update, context):
    start_time = time.process_time()
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = image_info.get_file()
    image_stream = BytesIO()

    update.message.reply_text(text='Хммм....дай подумать..', )

    image_file.download(out=image_stream)

    update.message.reply_text(text='Щас поглядимс..', )

    img = np.array(Image.fromarray(np.array(Image.open(image_stream))).resize((299, 299))).astype('float32') / 255.
    text = 'Я бы сказал, что это: \n'
    for i in range(10):
        text += ' '.join(generate_caption(img, t=5.)[1:-1]) + '\n'

    update.message.reply_text(text=text, parse_mode='Markdown')
    end_time = time.process_time()
    print('Duration of prediction: {}'.format(end_time - start_time))

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    updater = Updater(token=token,
                      base_url=None,
                      request_kwargs={'proxy_url': proxy},
                      use_context=True)
    dp = updater.dispatcher

    start_handler = CommandHandler("start", do_start)
    message_handler = MessageHandler(Filters.text, do_echo)
    photo_handler = MessageHandler(Filters.photo, send_prediction_on_photo)

    dp.add_handler(start_handler)
    dp.add_handler(message_handler)
    dp.add_handler(photo_handler)
    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":

    main()