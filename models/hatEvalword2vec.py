import string

import nltk
import numpy as np
import pandas as pd
from keras import Input
from keras.layers import Embedding, Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
import os
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.models import  Model
from keras.utils import np_utils, to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, libsvm

import  re
def removeStopwords(tweets):
    stopwords = nltk.corpus.stopwords.words("english")
    # stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    tokens = [tok for tok in tweets if not tok in stopwords]
    return tokens


def removeURL(tweets):
    # newText = re.sub('http\\S+', '', tweets, flags=re.MULTILINE)
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweets)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    # parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    return parsed_text


def removeNumber(tweets):
    newText = re.sub('\\d+', '', tweets)
    return newText


def removeHashtags(tokens):
    toks = [tok for tok in tokens if tok[0] != '#']
    #     if segment == True:
    #         segTool = Analyzer('en')
    #         for i, tag in enumerate(self.hashtags):
    #             text = tag.lstrip('#')
    #             segmented = segTool.segment(text)

    return toks


def removePunctuation(tweets):
    translator = str.maketrans('', '', string.punctuation)
    return tweets.translate(translator)


def stemTweet(tokens):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words


# In[32]:


def preprocess(tweet, remove_punctuation=True, remove_stopwords=False, remove_url=True, remove_hashtags=False,
               remove_number=True, stem_tweet=False):
    #     text = tweet.translate(string.punctuation)   -> to figure out what it does ?
    """
        Tokenize the tweet text using TweetTokenizer.
        set strip_handles = True to Twitter username handles.
        set reduce_len = True to replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    if remove_punctuation:
        tweet = removePunctuation(tweet)
    if remove_url:
        tweet = removeURL(tweet)
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)
    if remove_number:
        tweet = removeNumber(tweet)
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]
    if remove_hashtags:
        tokens = removeHashtags(tokens)
    if remove_stopwords:
        tokens = removeStopwords(tokens)
    if stem_tweet:
        tokens = stemTweet(tokens)
    text = " ".join(tokens)
    return text


# df = pd.read_csv('hateval2019-dataset/hateval2019_en_train.csv', quotechar="\"", encoding='utf-8')
df = pd.read_csv('gdrive/My Drive/GMU/hatEval-2019/public_development_en/hateval2019_en_train.csv',  quotechar="\"", encoding='utf-8')
df['text'] = df['text'].map(
    lambda x: preprocess(x, remove_stopwords=False, remove_hashtags=False, remove_number=True, remove_url=True,
                         stem_tweet=False))
y_train = to_categorical(df['HS'])

# df1 = pd.read_csv('hateval2019-dataset/hateval2019_en_test.csv', quotechar="\"", encoding='utf-8')
df1 = pd.read_csv('gdrive/My Drive/GMU/hatEval-2019/public_development_en/hateval2019_en_test.csv',  quotechar="\"", encoding='utf-8')
df1['text'] = df1['text'].map(
    lambda x: preprocess(x, remove_stopwords=True, remove_hashtags=False, remove_number=True, remove_url=True,
                         stem_tweet=False))
y_test = to_categorical(df1['HS'])



# train model
#
# model = Word2Vec(df['text'], min_count=1)
#
# print (model)


# =======================Convert string to index================
# Tokenizer

tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(df['text'].values)
# If we already have a character list, then replace the tk.word_index
# If not, just skip below part

# -----------------------Skip part start--------------------------
# construct a new vocabulary
# alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
# char_dict = {}
# for i, char in enumerate(alphabet):
#     char_dict[char] = i + 1


tokenizer_train = Tokenizer()
tokenizer_train.fit_on_texts(df['text'].values)

tokenizer_test = Tokenizer ()
tokenizer_test.fit_on_texts (df1['text'].values)

# -----------------------Skip part end----------------------------

# Convert string to index

train_sequences = tokenizer_train.texts_to_sequences(df['text'].values)
test_sequences = tokenizer_test.texts_to_sequences(df1['text'].values)

vocab_size = len(tokenizer_train.word_index) + 1
# Padding
train_data = pad_sequences(train_sequences, maxlen=140, padding='post')
test_data = pad_sequences(test_sequences, maxlen=140, padding='post')

# Convert to numpy array
train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')

print (train_data.shape)
print (test_data.shape)
# print (train_data)

# =======================Get classes================
train_classes = df['HS'].values
train_class_list = [x for x in train_classes]

test_classes = df1['HS'].values
test_class_list = [x for x in test_classes]

print (train_class_list, test_class_list)
print (train_classes)
from keras.utils import to_categorical

train_classes = to_categorical(train_class_list)
test_classes = to_categorical(test_class_list)




embeddings_dictionary = dict()
glove_file = open('hateval2019-dataset/glove.6B/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions

glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tk.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# model>>>>>>>>>>>>>>>>>>>>>

embedding_dim = 50
maxlen = 140
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 2
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

embedding_weights = []  # (70, 69)
embedding_weights.append(np.zeros(vocab_size))

for char, i in tokenizer_train.word_index.items():  # from index 1 to 69
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')

inputs = Input(shape=(maxlen,), name='input', dtype='int64')


embedding_layer = Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen)

x = embedding_layer(inputs)
# Conv
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
x = Flatten()(x)  # (None, 8704)
# Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
    x = Dropout(dropout_p)(x)
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)
# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
model.summary()

indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)

x_train = train_data[indices]
y_train = train_classes[indices]

x_test = test_data
y_test = test_classes

# Training
batch_size = 32
epochs = 20
model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, verbose=2)

y_predict = model.predict(x_test, batch_size=None, steps=None)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)



print("Precision\t", precision_score(y_test, y_predict, average=None))
print("Recall   \t", recall_score(y_test, y_predict, average=None))
print("F1-Score \t", f1_score(y_test, y_predict, average=None))
print("ROC-AUC  \t", roc_auc_score(y_test, y_predict, average=None))

target_names = ['0', '1']
print (classification_report (y_test, y_pred = y_predict, target_names=target_names))


