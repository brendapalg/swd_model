

# =============================================================================
# Imports
# =============================================================================

#Python modules
import numpy as np
import pandas as pd
from flask import Flask, request, Response, json
from sklearn.preprocessing import StandardScaler
# import plotly
# import plotly.express as px

# Load data
opinion_dict = pd.read_csv('../data/preprocessed/opinion_dict.csv')
bumble_data = pd.read_csv('../data/preprocessed/bumble_preprocessed.csv')
bumble_data.dropna(inplace=True)
bumble_data.reset_index(inplace=True, drop=True)

# To datetime
bumble_data['at'] = pd.to_datetime(bumble_data['at'])


# Create flask instance
app = Flask(__name__)

@app.route('/api', methods = ['GET', 'POST'])
def get_trends():
    trends = bumble_data.groupby(by=pd.Grouper(key="at", freq="M")).agg({"sentiment":"mean", "score": "mean", "content": "count"})
    scaler = StandardScaler()
    trends_norm = pd.DataFrame(scaler.fit_transform(trends.values))
    trends_norm.columns = trends.columns
    # trends.set_index(trends.index)
    trends_norm['date'] = trends.index.values
    
    # fig = px.line(trends_norm, x=trends_norm.index, y=['score', 'sentiment'],
    #           title='Bumble Review trends')
    return Response(trends_norm.to_json())

