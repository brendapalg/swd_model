

# =============================================================================
# Imports
# =============================================================================

#Python modules
import numpy as np
import pandas as pd
from flask import Flask, request, Response, json
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import gensim.corpora as corpora
import pyLDAvis.gensim_models
from gensim.test.utils import datapath
import gensim
import pickle 
import pyLDAvis
from flask import render_template


# =============================================================================
# Load Data
# =============================================================================
opinion_dict = pd.read_csv('../data/preprocessed/opinion_dict.csv')
bumble_data = pd.read_csv('../data/preprocessed/bumble_preprocessed.csv')
bumble_data.dropna(inplace=True)
bumble_data.reset_index(inplace=True, drop=True)

# To datetime
bumble_data['at'] = pd.to_datetime(bumble_data['at'])


# =============================================================================
# Topic Modeling
# =============================================================================
extra_stopwords = ['app']


# Create Dictionary
data_words = list(map(lambda x: [w for w in x.split() if w not in extra_stopwords], bumble_data['clean_content']))
id2word = corpora.Dictionary(data_words) 

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_words]

# Build LDA model
num_topics = 5
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

# Save model
model_path = "./models/ldaModel_topics-"+str(num_topics)
with open(model_path, 'wb') as f:
    pickle.dump(lda_model, f)


# Prepare visualization
LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

# Save visualization
vis_path = "./models/ldaVis_topics-"+str(num_topics)+".html"
pyLDAvis.save_html(LDAvis_prepared,vis_path)



# =============================================================================
# Create flask instance
# =============================================================================

app = Flask(__name__)

@app.route('/trends', methods = ['GET', 'POST'])
def get_trends():
    trends = bumble_data.groupby(by=pd.Grouper(key="at", freq="M")).agg({"sentiment":"mean", "score": "mean", "content": "count"})
    scaler = StandardScaler()
    trends_norm = pd.DataFrame(scaler.fit_transform(trends.values))
    trends_norm.columns = trends.columns
    trends_norm.set_index(trends.index, inplace=True)
    
    fig = px.line(trends_norm, x=trends_norm.index, y=['score', 'sentiment'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/topics', methods = ['GET', 'POST'])
def get_topics():
    return pyLDAvis.prepared_data_to_html(LDAvis_prepared)

