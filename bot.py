import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

new_model = tf.keras.models.load_model('my_model.h5')
data=pd.read_csv('label_class1.csv')
logging.basicConfig(filename="sample.log", level=logging.INFO)

def start(bot,update):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Send me bbox')
    


def help(bot,update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Send me image to predict as a photo')


def echo(bot,update):
    """Process the user image."""
    #Download image from user
    user = update.message.from_user
    logging.info(user)
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    #Preprocess it
    foto=Image.open('user_photo.jpg')
    foto = np.asarray( foto, dtype="float32")
    foto = (foto/127.5-1)
    image = tf.image.resize(foto, (83,45))
    image = (np.expand_dims(image,0))
    #Predict class label
    troika=new_model.predict_classes(image)
    update.message.reply_text('Predicted Class label is:')
    ans=data[data['label']==int(troika)]['class_'].item()
    update.message.reply_text(str(ans))
    
    
    



def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("UNIQUETOKEN")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.photo, echo))

    
    # Start the Bot
    updater.start_polling()

    
    updater.idle()


if __name__ == '__main__':
    main()









# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.

