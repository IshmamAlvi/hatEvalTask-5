import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
import os
import string
from keras.preprocessing.text import Tokenizer

from keras.utils import np_utils, to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, libsvm
from sklearn.ensemble import  RandomForestClassifier
import  re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
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

def convert_emojis(text):
    for emot in UNICODE_EMO:
            text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text



# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def preprocess(tweet, remove_punctuation=True, remove_stopwords=True, remove_url=True, remove_hashtags=False,
               remove_number=True, stem_tweet=True, emoji=True, emoticons = True):
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

    if emoji:
        text = convert_emojis(text)
    if emoticons:
        text = convert_emoticons(text)


    return text


df = pd.read_csv('hateval2019-dataset/hateval2019_es_train.csv', quotechar="\"", encoding='utf-8')
# df = pd.read_csv('gdrive/My Drive/GMU/hatEval-2019/public_development_en/hateval2019_en_train.csv',  quotechar="\"", encoding='utf-8')
# df = df.dropna()
# df = df.loc[df['HS'] == 1]
df['text'] = df['text'].map(
    lambda x: preprocess(x, remove_stopwords=False, remove_hashtags=False, remove_number=True, remove_url=True,
                         stem_tweet=False, emoji=True, emoticons=True))
y_train = to_categorical(df['HS'])

df1 = pd.read_csv('hateval2019-dataset/hateval2019_es_test.csv', quotechar="\"", encoding='utf-8')
# df1 = pd.read_csv('gdrive/My Drive/GMU/hatEval-2019/public_development_en/hateval2019_en_test.csv',  quotechar="\"", encoding='utf-8')
# df = df.dropna()
# df = df.loc[df['HS'] == 1]
df1['text'] = df1['text'].map(
    lambda x: preprocess(x, remove_stopwords=False, remove_hashtags=False, remove_number=True, remove_url=True,
                         stem_tweet=False, emoji=True, emoticons= True ))
y_test = to_categorical(df1['HS'])


# The maximum number of words to be used. (most frequent)
MAX_NUMBER_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 140
# This is fixed.
EMBEDDING_DIM = 100

TFIDF_FEATURES = 5000

tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df1['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

tfidf = TfidfVectorizer (ngram_range=(1,3),max_features= TFIDF_FEATURES)
X_train = tfidf.fit_transform(df['text'].values).toarray()
print ('Shape of tfidf train data tensor', X_train.shape, y_train.shape)

tfidf_test = TfidfVectorizer (ngram_range=(1,3),max_features= TFIDF_FEATURES)
X_test = tfidf_test.fit_transform(df1['text'].values).toarray()
print ('Shape of tfidf test data tensor', X_test.shape)


# get pos tags

def get_pos_tag(tweets):



    pos_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), max_features=TFIDF_FEATURES
        )

    tweet_tags = []
    for tweet in tweets:
        tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
        tokens = tweet.split()
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    return np.array(pos)


from textstat.textstat import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment_analyzer = VS()
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    # twitter_objs = count_twitter_objs(tweet)
    # retweet = 0
    # if "rt" in words:
    #     retweet = 1
    # features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
    #             num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound']
    #            ]
    features = [  avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms
                ]
    # features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

pos = get_pos_tag(df['text'])
features = get_feature_array(df['text'])
M = np.concatenate([X_train, features],axis=1)


pos1 = get_pos_tag(df1['text'])
features1 = get_feature_array(df1['text'])

print (X_test.shape, pos.shape, pos1.shape)
# M_test = np.concatenate([X_test, pos1[:,None]],axis=1)
M_test = np.concatenate([X_test, features1],axis=1)

X_train = pd.DataFrame (M)
X_test = pd.DataFrame (M_test)

print ('others:' , X_train.shape, X_test.shape)

y = df['HS'].astype(int)
y_test = df1['HS'].astype(int)

print (y.shape)

####################### model building ###################
# select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
# X_train = select.fit_transform(X_train,y_train)
#
# print (X_train.shape)

# model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr').fit(X_train, y)

# model = LogisticRegression(class_weight='balanced',penalty='l2',C=0.01).fit(X_train,y)


#
# pipe = Pipeline(
#         [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
#                                                   penalty="l1", C=0.1))),
#         ('model', LogisticRegression(class_weight='balanced',penalty='l2'))])


#
pipe = Pipeline(
        [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                  penalty="l1", C=0.01))),
        ('model', LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr'))])




# pipe = Pipeline(
#         [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
#                                                   penalty="l1", C=0.1))),
#         ('model', RandomForestClassifier(n_estimators=100, max_depth= 10))])


# param_grid = [{'C': [1, 5], 'kernel': ('linear', 'rbf')}] # optional params
param_grid = [{}]
grid_search = GridSearchCV(pipe,
                           param_grid,
                           cv=StratifiedKFold(n_splits=10,
                                              random_state=42).split(X_train, y),verbose=2)

model = grid_search.fit(X_train, y)

# X_test = np.array(X_test)
# X_test.reshape(3000,10000)

train, X_test = X_train.align(X_test, join='outer', fill_value= -1, axis=1)
X_test.fillna(X_test.mean())

y_predict = model.predict (X_test)

report = classification_report(y_true= y_test, y_pred= y_predict)

print (report)


print("Precision\t", precision_score(y_test, y_predict, average=None))
print("Recall   \t", recall_score(y_test, y_predict, average=None))
print("F1-Score \t", f1_score(y_test, y_predict, average=None))
print("ROC-AUC  \t", roc_auc_score(y_test, y_predict, average=None))



