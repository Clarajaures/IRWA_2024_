import os
from json import JSONEncoder

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/farmers-protest-tweets.json"
print(file_path)

# file_path = "../../tweets-data-who.json"
corpus = load_corpus(file_path)
print("loaded corpus. first elem:", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2021 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    analytics_data.save_session_terms(user_ip,agent)
    print(session)

    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus)
    
    print("THIS IS THE FIRST RESULT",results[0])


    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')

    print("doc details session: ")
    print(session)

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]
    p1 = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))

    twt = list(corpus.values())[p1]
    #print("The document title is:", twt)

    doc = Document(
        id=twt["id"],
        title=twt["title"],
        description=twt["description"],
        doc_date=twt["doc_date"],
        likes=twt["likes"],
        retweets=twt["retweets"],
        url=twt["url"],
        hashtags=twt["hashtags"]
    )
    #clicked_tweet = Document(p1, twt[1], twt[2], twt[3], twt[4], twt[5], twt[6], twt[7])



    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1
    analytics_data.save_clicks()

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))

    return render_template('doc_details_2.html', Tweet_result = doc)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)
    # ### End replace with your code ###


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]  # Fetch document details from corpus
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])  # Create ClickedDoc object
        visited_docs.append(doc)

    # Sort visited_docs by the click count (counter)
    visited_docs = [
        {"doc_id": doc_id, **data} if isinstance(data, dict) else {"doc_id": doc_id, "counter": data, "description": "N/A"}
        for doc_id, data in analytics_data.fact_clicks.items()
    ]

    total_clicks = sum([doc["counter"] for doc in visited_docs])
    total_queries = len(analytics_data.fact_query_terms)

    return render_template(
        'dashboard.html',
        visited_docs=visited_docs,
        fact_query_terms=analytics_data.fact_query_terms,
        fact_two=analytics_data.fact_two,
        fact_three=analytics_data.fact_three,
        total_clicks=total_clicks,
        total_queries=total_queries
    )


@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
