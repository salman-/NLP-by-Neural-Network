import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization,Embedding,Input,GlobalAveragePooling1D,Dense,LSTM,GRU,Bidirectional,Conv1D,GlobalMaxPool1D
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

# Download data (same as from Kaggle)
#!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
zip_ref = zipfile.ZipFile("nlp_getting_started.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df['text'],
                                                                            train_df['target'],
                                                                            test_size=.2,
                                                                            shuffle=True)

def count_words_in_sentence(sentence):
    return len(sentence.split(" "))

x = train_df['text'].apply(count_words_in_sentence)
average_output_sequence_length = int(np.round(np.mean(x)))

text_vectorization_layer =  TextVectorization(max_tokens=10000,
                                              ngrams=10,
                                              standardize='lower_and_strip_punctuation',
                                              output_mode='int',
                                              output_sequence_length = 15
                                              )
text_vectorization_layer.adapt(['BlackBerry Limited is a Canadian software','Hello baby'])
text_vectorization_layer(['BlackBerry Limited is a Canadian software','Hello baby'])

text_vectorization_layer.get_vocabulary()

token_indices = text_vectorization_layer(['BlackBerry Limited is a Canadian software']).numpy()[0]

#-------------------------------------------------------
# Get vocabs
ngram_tokens = text_vectorization_layer.get_vocabulary()

text_vector = []
# Specify each index is referring to which vocab
for i in token_indices:
    #print("index:", i, "| vocab: ", ngram_tokens[i])
    text_vector.append(ngram_tokens[i])

text_vector

embeding = Embedding(input_dim=len(ngram_tokens),embeddings_initializer='uniform',output_dim=128, name= 'embeding_layer')
embeding(token_indices)

#--------------------------------------------------

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = Embedding(input_dim=len(ngram_tokens),embeddings_initializer='uniform',output_dim=128, name= 'embeding_layer')(x)
x = GlobalAveragePooling1D()(x)
output = Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(input,output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'] )

model.fit(x=train_sentences,y=train_labels,epochs=6,validation_data=(val_sentences,val_labels))

#-------------------------------------------------------

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = embeding(x)
x = LSTM(64)(x)
output = Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(input,output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'] )

model.fit(x=train_sentences,y=train_labels,epochs=6,validation_data=(val_sentences,val_labels))

#----------------------------------------------
# GRU NN
input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = embeding(x)
x = GRU(64)(x)
output = Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(input,output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'] )

model.fit(x=train_sentences,y=train_labels,epochs=6,validation_data=(val_sentences,val_labels))

#---------------------------------------------
# Bidirectional

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = embeding(x)
x = Bidirectional(LSTM(64))(x)
output = Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(input,output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'] )

model.fit(x=train_sentences,y=train_labels,epochs=6,validation_data=(val_sentences,val_labels))

#-----------------------------------------
# Conv1D

input = Input(shape=(1,),dtype='string')
x = text_vectorization_layer(input)
x = embeding(x)
x = Conv1D(filters=32, kernel_size=3 ,activation='relu')(x)
x = GlobalMaxPool1D()(x)
output = Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(input,output)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'] )

model.fit(x=train_sentences,y=train_labels,epochs=6,validation_data=(val_sentences,val_labels))

