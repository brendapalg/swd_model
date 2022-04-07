

# =============================================================================
# Imports
# =============================================================================

#Python modules
import numpy as np
import pandas as pd
from flask import Flask, json
from sklearn.preprocessing import StandardScaler
import plotly
import plotly.express as px
import pyLDAvis
import pickle 


# =============================================================================
# Load Data
# =============================================================================
bumble_data = pd.read_csv('./data/preprocessed/bumble_preprocessed.csv')
bumble_data.dropna(inplace=True)
bumble_data.reset_index(inplace=True, drop=True)

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

