import pandas as pd
import json
import os
from collections import defaultdict
from array import array
import math
import numpy as np



def create_index(lines):
    index = defaultdict(list)
    title_index = {}    # dictionary to map words to tweet ids
    for line in lines:  # Remember, lines contain all tweets: tweet_id, tweet_text, tweet_date, hashtags, tweet_likes, tweet_retweets, url

        tweet_id = int(line[0])
        tweet_text = line[1]
        #print(tweet_text)

        ## ===============================================================

        current_page_index = {}

        for position, term in enumerate(tweet_text):       # terms contains page_title + page_text. Loop over all terms
            if term in current_page_index:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list
                current_page_index[term][1].append(position)

            else:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[tweet_id, array('I',[position])]     #'I' indicates unsigned int (int in Python)

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        ## END CODE

    return index


def create_index_tfidf(lines, num_documents):

    index = defaultdict(list)
    tf = defaultdict(lambda: defaultdict(float))    # Term frequencies: tf = { "term1": {tweet_id1: freq1, tweet_id2: freq1, …}, "term2": {tweet_idx: freqx,…}, …}
    df = defaultdict(int)                           # Document frequencies: df = { "term1": #unique_docs_where_term1_appears, "term2": #unique_docs_where_term2_appears, …}
    title_index = defaultdict(str)
    idf = defaultdict(float)

    for line in lines:
        tweet_id = int(line[0])                 # Tweet ID
        tweet_text = line[1]                    # Tweet text
        title_index[tweet_id] = tweet_text      # Map tweet_id to tweet_text

        terms = tweet_text                      # Already tokenized
        current_page_index = {}

        # Build the index for the current tweet
        for position, term in enumerate(terms):
            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [tweet_id, array('I', [position])]

        # Compute normalization factor (norm) for TF in this tweet
        norm = math.sqrt(sum(len(posting[1]) ** 2 for posting in current_page_index.values()))

        # Calculate TF for terms in the current tweet and update DF
        for term, posting in current_page_index.items():
            # tf[term].append(round(len(posting[1]) / norm, 4))  # Normalized TF
            tf[term][tweet_id] = round(len(posting[1]) / norm, 4)
            df[term] += 1  # Document frequency count for this term

        # Merge current tweet's index with main index
        for term, posting in current_page_index.items():
            index[term].append(posting)

    # Calculate IDF for each term
    for term, freq in df.items():
        idf[term] = round(np.log(float(num_documents) / freq), 4)

    return index, tf, df, idf, title_index
