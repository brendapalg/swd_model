

# =============================================================================
# Imports
# =============================================================================

#Python modules
import numpy as np
import pandas as pd
from flask import Flask, json, request, Response
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import pyLDAvis
import pickle 


# =============================================================================
# Load Data
# =============================================================================
bumble_data = pd.read_csv('./data/preprocessed/bumble_preprocessed.csv')

# To datetime
bumble_data['at'] = pd.to_datetime(bumble_data['at'])


# =============================================================================
# Topic Modeling
# =============================================================================
num_topics = 5
model_path = "./models/ldaModelvis_topics-"+str(num_topics)

with open(model_path, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

# Save visualization
vis_path = "./models/ldaVis_topics-"+str(num_topics)+".html"
pyLDAvis.save_html(LDAvis_prepared,vis_path)


# =============================================================================
# Sample by sentiment and date
# =============================================================================

def sample_reviews(df, date, sentiment, n=3):
    if sentiment == 'pos':
        sample = df.loc[(df['my']==date) & (df['sentiment']>0), 'content'].sample(n).values
    else:
        sample = df.loc[(df['my']==date) & (df['sentiment']<0), 'content'].sample(n).values
    return list(sample)



# =============================================================================
# Create flask instance
# =============================================================================

app = Flask(__name__)

@app.route('/trends', methods = ['GET', 'POST'])
def get_trends():
    # Get plot of sentiment trend
    trends = bumble_data.groupby(by=pd.Grouper(key="at", freq="M")).agg({"sentiment":"mean", "score": "mean", "content": "count"})
    scaler = StandardScaler()
    trends_norm = pd.DataFrame(scaler.fit_transform(trends.values))
    trends_norm.columns = trends.columns
    trends_norm.set_index(trends.index, inplace=True)

    if request.method == 'POST':
        data = request.get_json(force=True)
        date = data["date"]
        
        pos = sample_reviews(bumble_data, date, 'pos')
        neg = sample_reviews(bumble_data, date, 'neg')
    else:
        pos = ""
        neg = ""

    
    fig = px.line(trends_norm, x=trends_norm.index, y=['score', 'sentiment'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    resp = json.dumps({'graphJSON': graphJSON, 'pos': pos, 'neg': neg})

    return Response(resp)

@app.route('/topics', methods = ['GET', 'POST'])
def get_topics():
    return pyLDAvis.prepared_data_to_html(LDAvis_prepared)

