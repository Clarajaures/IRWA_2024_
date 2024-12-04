import pandas as pd
import json
from myapp.search.objects import Document
from pandas import json_normalize

from myapp.core.utils import load_json_file
from myapp.search.objects import Document
import nltk

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download the missing 'punkt_tab' data

_corpus = []
stop_words = set(stopwords.words('english'))    #define the stop words in english
stemmer = PorterStemmer()                       #initialize the Porter Stemmer
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''    #define the punctuation we want to remove

def preprocess_text(tweet):
    # Step 1: Remove URLs, hashtags, mentions, and non-alphabetical characters
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)   # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)                      # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)                      # Remove hashtags
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)               # Remove non-alphabetical character

    # ADDITIONALLY
    tweet = tweet.lower()                                   # As additional pre-processing step
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)             # Remove non-ASCII characters(emojis)
    tweet = ''.join([char for char in tweet if char not in punctuation])
    words = word_tokenize(tweet)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    cleaned_tweet = stemmed_words
    return cleaned_tweet

def extract_tweets(text):
    tweet_id = text.get("id","N/A")
    tweet_text = text.get("content","N/A")
    tweet_date = text.get("date", "N/A")
    hashtags = [hashtag[1:] for hashtag in re.findall(r'#\S+', tweet_text)]
    tweet_likes = text.get("likeCount", 0)
    tweet_retweets = text.get("retweetCount", 0)
    url = f"https://twitter.com/{text['user']['username']}/status/{text['id']}"

    #after creating the hashtags we pre-process the content
    tweet_text = preprocess_text(tweet_text)

    return [tweet_id, tweet_text, tweet_date, hashtags, tweet_likes, tweet_retweets, url]

def load_corpus(path) -> [Document]:
    with open(path, 'r') as fp:
        for line in fp:
            try:
                tweet_data = json.loads(line)
                formatted_tweet = extract_tweets(tweet_data)
                _corpus.append(formatted_tweet)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    clean_tweets_dict = {}
    clean_tweets_dict = {tweet[0]: tweet[1] for tweet in _corpus}

    return _corpus, clean_tweets_dict



# def _row_to_doc_dict(row: pd.Series):
#     _corpus[row['id']] = Document(row['id'], row.get('quote', ''), row.get('author', ''))
    
def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'],
                                  row['Retweets'],
                                  row['Url'], row['Hashtags'])
                                      

'''    
import pandas as pd
import json

from myapp.core.utils import load_json_file
from myapp.search.objects import Document
import nltk

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download the missing 'punkt_tab' data
import re


def _load_corpus_as_dataframe(path):
    """
    Load documents corpus from file in 'path'
    :return:
    """
    json_data = load_json_file(path)
    tweets_df = _load_tweets_as_dataframe(json_data)
    _clean_hashtags_and_urls(tweets_df)
    # Rename columns to obtain: Tweet | Username | Date | Hashtags | Likes | Retweets | Url | Language
    corpus = tweets_df.rename(
        columns={"id": "Id", "full_text": "Tweet", "screen_name": "Username", "created_at": "Date",
                 "favorite_count": "Likes",
                 "retweet_count": "Retweets", "lang": "Language"})

    # select only interesting columns
    filter_columns = ["Id", "Tweet", "Username", "Date", "Hashtags", "Likes", "Retweets", "Url", "Language"]
    corpus = corpus[filter_columns]
    return corpus


def _load_tweets_as_dataframe(json_data):
    data = pd.DataFrame(json_data).transpose()
    # parse entities as new columns
    data = pd.concat([data.drop(['entities'], axis=1), data['entities'].apply(pd.Series)], axis=1)
    # parse user data as new columns and rename some columns to prevent duplicate column names
    data = pd.concat([data.drop(['user'], axis=1), data['user'].apply(pd.Series).rename(
        columns={"created_at": "user_created_at", "id": "user_id", "id_str": "user_id_str", "lang": "user_lang"})],
                     axis=1)
    return data


def _build_tags(row):
    tags = []
    # for ht in row["hashtags"]:
    #     tags.append(ht["text"])
    for ht in row:
        tags.append(ht["text"])
    return tags


def _build_url(row):
    url = ""
    try:
        url = row["entities"]["url"]["urls"][0]["url"]  # tweet URL
    except:
        try:
            url = row["retweeted_status"]["extended_tweet"]["entities"]["media"][0]["url"]  # Retweeted
        except:
            url = ""
    return url


def _clean_hashtags_and_urls(df):
    df["Hashtags"] = df["hashtags"].apply(_build_tags)
    df["Url"] = df.apply(lambda row: _build_url(row), axis=1)
    # df["Url"] = "TODO: get url from json"
    df.drop(columns=["entities"], axis=1, inplace=True)


def load_tweets_as_dataframe2(json_data):
    """Load json into a dataframe

    Parameters:
    path (string): the file path

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """
    # Load the JSON as a Dictionary
    tweets_dictionary = json_data.items()
    # Load the Dictionary into a DataFrame.
    dataframe = pd.DataFrame(tweets_dictionary)
    # remove first column that just has indices as strings: '0', '1', etc.
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    return dataframe


def load_tweets_as_dataframe3(json_data):
    """Load json data into a dataframe

    Parameters:
    json_data (string): the json object

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """

    # Load the JSON object into a DataFrame.
    dataframe = pd.DataFrame(json_data).transpose()

    # select only interesting columns
    filter_columns = ["id", "full_text", "created_at", "entities", "retweet_count", "favorite_count", "lang"]
    dataframe = dataframe[filter_columns]
    return dataframe


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Id']] = Document(row['Id'], row['Tweet'][0:100], row['Tweet'], row['Date'], row['Likes'],
                                  row['Retweets'],
                                  row['Url'], row['Hashtags'])
                                  
                                  
                                  '''