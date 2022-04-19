import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Custom functions
os.chdir("..")
import utils.text_utils as tu


data = pd.read_csv("./data/raw/bumble_google_play_reviews.csv")

# Format columns
data['at'] = pd.to_datetime(data['at'] )
data['score'] = pd.to_numeric(data['score'] )

# Filter by date
data = data[data['at']>'2015-12-01']

# Select columns and filter out rows with missing values
columns = ['at', 'score', 'content']
clean_data = data.loc[~data.content.isna(),columns].copy()
print("Selected columns:", columns)
print("Rows dropped:", len(data)-len(clean_data))

# Clean text
clean_data['clean_content'] = list(map(lambda x: tu.clean_text(str(x), lemmatize=True), clean_data['content']))

# Save preprocessed data
clean_data.to_csv("./data/preprocessed/bumble_preprocessed.csv", index = False)

# Top words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(clean_data['clean_content'].values)