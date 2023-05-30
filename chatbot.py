
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
data = json.loads(open('data.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotexe.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
        return np.array(bag)
    
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model_predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.30
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'data': classes[r[0]], 'probability': str(r[1])})
   return return_list

def get_response(data list, data json):
        tag = data_list[0]['intent']
        list_of_data = data_json['data']
    for i in list_of_data:
     if i['tag'] == tag:
     result = random.choice(i['responses'])
     break
return result

print("Bot is running!")

while True:
     message = input("")
     ints = predict_class(message)
     res = get_response(ints, data)
     print(res)

