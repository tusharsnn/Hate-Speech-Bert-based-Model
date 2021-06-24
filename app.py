from flask import Flask, render_template,request
import numpy as np
# import pandas as pd
# import scipy
import tensorflow as tf
import tensorflow_text as text
import os
# import shutil
# import tensorflow_hub as hub
# import official.nlp.bert.tokenization as tokenization
import matplotlib.pyplot as plt
# tf.saved_model.LoadOptions(
#     experimental_io_device="/job:localhost"
# )

app = Flask(__name__)
# encoder_layer = hub.KerasLayer("My_model1/")
# vocab_file = encoder_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = encoder_layer.resolved_object.do_lower_case.numpy()
# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# def encode_names(n, tokenizer):
#    tokens = list(tokenizer.tokenize(n[0]))
#    tokens.append('[SEP]')
#    return tokenizer.convert_tokens_to_ids(tokens)

# def bert_encode(string_list, tokenizer, max_seq_length = 512):
#   num_examples = len(string_list)
  
#   string_tokens = tf.ragged.constant([
#       encode_names(n, tokenizer) for n in np.array(string_list)])

#   cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*string_tokens.shape[0]
#   input_word_ids = tf.concat([cls, string_tokens], axis=-1)
#   input_mask = tf.ones_like(input_word_ids).to_tensor(shape=(None, max_seq_length))

#   type_cls = tf.zeros_like(cls)
#   type_tokens = tf.ones_like(string_tokens)
#   input_type_ids = tf.concat(
#       [type_cls, type_tokens], axis=-1).to_tensor(shape=(None, max_seq_length))

#   inputs = {
#       'input_word_ids': input_word_ids.to_tensor(shape=(None, max_seq_length)),
#       'input_mask': input_mask,
#       'input_type_ids': input_type_ids}

#   return inputs

model = tf.keras.models.load_model("my_model/")


@app.route("/login", methods=['GET', 'POST'])
def login():
    tweets = ""
    label = ""
    if len(request.form)!=0:
        tweets = request.form["tweet"]
        # processed_tweet = bert_encode([[tweets]], tokenizer)
        prediction = np.argmax(model.predict([tweets]),axis=1)
        print(tweets)
        print(prediction)
        print(model.predict([tweets]))

        if prediction == 1:
            label = "Normal"
        else:
            label = "Offensive"
    return render_template( "login.html" , tweet=tweets, prediction = label)

# @app.route("/", methods=['GET', 'POST'])
# def index():
#     tweet = request.form["tweet"]
#     password = request.form["password"]
#     return render_template( "login.html" , hello = 32, tweet=tweet, prediction = prediction)


app.run(debug = True)