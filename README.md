# NLP-by-Neural-Network

In this project, we try to build an NLP neural-network which can predict if a Tweet is about a DISASTER or not !

## Dataset:
https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip

## Basic concepts in NLP

The ***Tokenization*** and ***Embedding*** are the main two concepts in NLP:

### Tokenization
Tokenization means presenting text by a (fixed length) vector of numbers.
The preprocessing layer (`tensorflow.keras.layers.TextVectorization`) in Tensorflow tokenize the text for us.
For example, this layer receives a list of Texts and provides an array of `Integer`.

Here is the steps to Tokenize a given text in word level:

1. Remove the punctuations
2. Convert the text to lowercase
3. Create NGRAM sequences of tokens (keep reading to figure out about NGRAM concept)
4. Consider each sequence as a vocab


### What is NGRAM ?
NGRAMs are collection of tokens which are produced by a sliding window over the text and captures some words.
This window starts from beginning and everytime it considers the words which are inside the window as a token (vocab) and then the window slides to the next word until the very end of the text.
Of course the NGRAM's window can have different size. If the window length is 1 word, then in every slide it collects only 1 single word.
If the NGRAM==2 then the length of the window is 2 words and it captures 2 words in each slide.

Let's assume we would like to create sequences of `NGRAM=1` for a given text like:

***BlackBerry Limited is a Canadian software***

The following video presents how the tokens with different ngrams (from 1 to 3) are generated.

![NGramm](https://user-images.githubusercontent.com/4312244/154011999-9d3c63d6-dbc9-4ced-954b-8293fbd58891.gif)
In Tensorflow framework, we can use the following code to tokenize a text:

<pre>
from tensorflow.keras.layers import TextVectorization
text_vectorization_layer =  TextVectorization(max_tokens=10000,
                                              ngrams=5, # Collect all the ngram tokens from 1 to 5
                                              standardize='lower_and_strip_punctuation',
                                              output_mode='int', # We present each token by its index in the vocabulary
                                              output_sequence_length = 5 # Fix length for out put
                                              )
text_vectorization_layer.adapt(['BlackBerry Limited is a Canadian software','Hello World'])
text_vectorization_layer(['BlackBerry Limited is a Canadian software','Hello World'])

#Output is:

< tf.Tensor: shape=(2, 15), dtype=int64, numpy=
array([[20,  7, 11, 24, 15,  2, 19,  6, 10, 23, 14, 18,  5,  9, 22],
       [13, 21, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])>
</pre>

The above code will create a group of tokens (called vocabulary) and we can get these vocabs by calling the below code:

<pre>
text_vectorization_layer.get_vocabulary()

# output is:
['',
 '[UNK]',
 'software',
 'limited is a canadian software',
 'limited is a canadian',
 'limited is a',
 'limited is',
 'limited',
 'is a canadian software',
 'is a canadian',
 'is a',
 'is',
 'hello baby',
 'hello',
 'canadian software',
 'canadian',
 'blackberry limited is a canadian',
 'blackberry limited is a',
 'blackberry limited is',
 'blackberry limited',
 'blackberry',
 'baby',
 'a canadian software',
 'a canadian',
 'a']
</pre>

It means that, the sentence like

`BlackBerry Limited is a Canadian software`

is now mapped to a vector such as below:

`[20,  7, 11, 24, 15,  2, 19,  6, 10, 23, 14, 18,  5,  9, 22]`

If we need to know what does each number mean then we can execute the below code:

<pre>
token_indices = text_vectorization_layer(['BlackBerry Limited is a Canadian software']).numpy()[0]

# Get vocabs
ngram_tokens = text_vectorization_layer.get_vocabulary()

text_vector = []
# Specify each index is referring to which vocab
for i in token_indices:
  #print("index:", i, "| vocab: ", ngram_tokens[i])
  text_vector.append(ngram_tokens[i])

text_vector
</pre>


## Embedding

Embedding is an approach to present the meaning of the text's features (extracted tokens).
We can define **"the meaning of each word"** by a 300 dimension vector and for each dimention we can have a value from -1 to 1. 
The higher the value, there is more connection in between the that token and that specific feature.
For example, in the below table we defined the meaning of the two words such as  "Laptop" and "Glasses".
The vertical column is a sample of "dimensions" which can define the words.
As the word "Laptop" has more connection to the features such as "Technology", "Device" and "Entertainment", 
it receives higher values in these features. However, laptop got very low scores to the other dimensions such as "Food" or "Body"

![Embedding matrix](https://user-images.githubusercontent.com/4312244/154244418-d333fc53-4341-45c9-8226-7247331e9ba5.png)

The Tensorflow can generate the embedding matrix and improve the values in each itteration.

<pre>
from tensorflow.keras.layers import Embedding
embeding = Embedding(input_dim=len(ngram_tokens),  # Number of extracted vocabularies
                     embeddings_initializer='uniform', 
                     output_dim=128,
                     name= 'embeding_layer')
                     
embeding(token_indices)
</pre>

This layer creates an embedding matrix (`input_dim*output_dim`) and after each itteration the values of this matrix gets updated.

Also, we can use `Transfer-learning` using the following code:

<pre>
embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                 input_shape=[], # shape of inputs coming to our model 
                                 dtype=tf.string, # data type of inputs coming to the USE layer
                                 trainable=False, # keep the pretrained weights (we'll create a feature extractor)
                                 name="USE")
</pre>
