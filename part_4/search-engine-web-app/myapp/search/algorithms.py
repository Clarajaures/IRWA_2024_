from collections import defaultdict
from array import array
import numpy as np
import math
import collections
import numpy.linalg as la

from myapp.search.load_corpus import preprocess_text
from myapp.search.objects import ResultItem, Document

# Our function of the previous labs
def create_index_tfidf(lines, num_documents):
    """
    Build an inverted index and compute tf, df, and idf.

    Arguments:
    lines -- collection of tweets, where each tweet is a line formatted as tweet_id|tweet_text|tweet_date|hashtags|tweet_likes|tweet_retweets|url
    num_documents -- total number of tweets (documents)

    Returns:
    index - inverted index with terms as keys and lists of document ids and positions
    tf - normalized term frequency for each term in each document
    df - document frequency for each term
    idf - inverse document frequency for each term
    title_index - maps tweet_id to tweet_text for display in search results
    """

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

def rank_documents(terms, docs, index, idf, tf, title_index):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Ranked documents and the scores associated
    """

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would become 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc] * idf[term]  # TODO: check if multiply for idf

    # Calculate the score of each doc
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)

    result_docs = [x[1] for x in doc_scores]

    # #print document titles instead if document id's
    # if len(result_docs) == 0:
    #     print("No results found, try again")
    #     query = input()
    #     docs = search_tf_idf(query, index, idf, tf, title_index)
    #print ('\n'.join(result_docs), '\n')

    return result_docs, doc_scores

def search_tf_idf(query, index, idf, tf, title_index):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    term_docs_search = []
    # print(f"Original Query: {query}")
    query = preprocess_text(query)        # we use the same function to preprocess the queries(as the tweets) -> returns tokens
    # print(f"Preprocessed Query: {query}")

    docs = None
    for term in query:
        if term in index:
            term_docs_search = [posting[0] for posting in index[term]]
            term_docs_set = set(term_docs_search)

            # Intersect docs with term_docs_set only if docs is not None
            if docs is None:
                docs = term_docs_set  # Initialize docs with the first term's results
            else:
                docs &= term_docs_set  # Intersection to keep only docs containing all terms
                
    docs = list(docs)

    return docs

def search_in_corpus(corpus, query, search_id):
    # 1. create create_tfidf_index
    docs_corpus = list(corpus.values())
    num_documents = len(docs_corpus)
    
    index, tf, df, idf, title_index = create_index_tfidf([(doc.id, doc.title) for doc in docs_corpus], num_documents)
    
    index_5_items = list(index.items())[:5]
    print("Indexed terms:", list(index.keys())[:10])

    tf_5_items = list(tf.items())[:5]
    # print('tf: {}'.format(tf_5_items))

    df_5_items = list(df.items())[:5]
    print('df: {}'.format(df_5_items))

    idf_5_items = list(idf.items())[:5]
    print('idf: {}'.format(idf_5_items))

    title_5_items = list(title_index.items())[:5]
    print('title: {}'.format(title_5_items))
    
    docs_ids = search_tf_idf(query, index, idf, tf, title_index)
    #print(f"Document IDs from search: {docs_ids}")
    
    if not docs_ids:
        print("No results found, try again.")
        return [], []
        
    # 2. apply ranking
    ranked_docs, doc_scores = rank_documents(preprocess_text(query), docs_ids, index, idf, tf, title_index)
    
    # 3. Construct the list of ResultItem objects
    res = []
    for score, doc_id in doc_scores:
        item = corpus.get(doc_id)
        if not item:
            continue  # Skip if metadata for the document is missing

        res.append(ResultItem(
            id=item.id,
            title=item.title,
            description=item.description,
            doc_date=item.doc_date,
            url="doc_details?id={}&search_id={}&param2=2".format(item.id, search_id),
            ranking=score
        ))

    return res
