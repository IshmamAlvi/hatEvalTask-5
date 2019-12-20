from keras.callbacks import EarlyStopping
from keras_preprocessing import sequence
import re

from partd import numpy
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
import os


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, GRU, Input, \
    Bidirectional, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.utils import np_utils, to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
    
    
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
    toks = [ tok for tok in tokens if tok[0] != '#']
#     if segment == True:
#         segTool = Analyzer('en')
#         for i, tag in enumerate(self.hashtags):
#             text = tag.lstrip('#')
#             segmented = segTool.segment(text)

    return toks

def removePunctuation (tweets):
    translator = str.maketrans('', '', string.punctuation)
    return tweets.translate(translator)


def stemTweet(tokens):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words

# In[32]:


def preprocess(tweet, remove_punctuation = True, remove_stopwords = False, remove_url = True, remove_hashtags = False, remove_number = True, stem_tweet = False):
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

df = pd.read_csv('hateval2019-dataset/hateval2019_en_train.csv',  quotechar="\"", encoding='utf-8')
# df = pd.read_csv('gdrive/My Drive/GMU/hatEval-2019/public_development_en/hateval2019_en_train.csv',  quotechar="\"", encoding='utf-8')
# df = df.dropna()
# df = df.loc[df['HS'] == 1]
df['text'] = df['text'].map(lambda x: preprocess(x, remove_stopwords=True, remove_hashtags=False, remove_number= True, remove_url=True, stem_tweet=False))
y_train = to_categorical(df['HS'])

df1 = pd.read_csv('hateval2019-dataset/hateval2019_en_test.csv',  quotechar="\"", encoding='utf-8')
# df1 = df1.dropna()
# df1 = df1.loc[df1['HS'] == 1]
df1['text'] = df1['text'].map(lambda x: preprocess(x, remove_stopwords=True, remove_hashtags=False, remove_number= True, remove_url=True, stem_tweet=False))
y_test = to_categorical(df1['HS'])



# The maximum number of words to be used. (most frequent)
MAX_NUMBER_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 140
# This is fixed.
EMBEDDING_DIM = 100


tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df1['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



X_train = tokenizer.texts_to_sequences(df['text'].values)
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train data tensor:', X_train.shape, y_train.shape)

X_test = tokenizer.texts_to_sequences(df1['text'].values)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test data tensor:', X_test.shape, y_test.shape)



################### model building###################3
model = Sequential()
model.add(Embedding(MAX_NUMBER_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense (64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print (model.summary())
epochs = 15
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

y_predict = model.predict(X_test, batch_size=None, steps=None)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)



print("Precision\t", precision_score(y_test, y_predict, average=None))
print("Recall   \t", recall_score(y_test, y_predict, average=None))
print("F1-Score \t", f1_score(y_test, y_predict, average=None))
print("ROC-AUC  \t", roc_auc_score(y_test, y_predict, average=None))

target_names = ['0', '1']
print (classification_report (y_test, y_pred = y_predict, target_names=target_names))
f = open ("GRUModel/hatEvalGRUEN_HS.txt","w")
f.write(classification_report(y_test, y_pred = y_predict, target_names=target_names))
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('GRUModel/epochVsAccEN_HS.pdf')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('GRUModel/epochVslossEN_HS.pdf')
plt.show()

#############heat map############

import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix

seaborn.countplot(df['HS'])
plt.xlabel('Label')
plt.title('Classification')
plt.show()

confusion_matrix = confusion_matrix(y_test , y_predict)
acc = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
print('Overall accuracy: {} %'.format(acc*100))
matrix_proportions = np.zeros((2, 2))
for i in range(0, 2):
    matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
names = ['Non AG', 'AG']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(10, 10))
seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 20}, cbar=False, square=True, fmt='.2f', cmap="RdBu_r")
plt.ylabel(r'True categories', fontsize=25)
plt.xlabel(r'Predicted categories', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='minor', labelsize=10)
# #Uncomment line below if you want to save the output
plt.savefig('GRUModel/ConfusionMatrixGRUEN_HS.pdf')





