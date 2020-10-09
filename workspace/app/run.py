import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_Categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Visual 1 - Distribution of messages across the three genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visual 2 - Messages in each category
    categories = df.drop(['id','message','original','genre'], axis=1)
    categories.dropna(inplace=True)
    category_totals = categories.sum(axis=0)

    # Visual 3 - Seeing what types of Weather issue cause a food shortage as well
    food_weather_cols = ['food','floods','storm','fire','earthquake','cold','other_weather']
    food_categories = df[food_weather_cols]
    food_related = food_categories[food_categories['food']==1.0]
    weather_totals = food_related.sum(axis=0)
    food_total = weather_totals.pop('food')
    weather_percentages = weather_totals*100/food_total

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=category_totals.index,
                    y=category_totals.values
                )
            ],

            'layout': {
                'title': 'Messages in each category',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=weather_percentages.index,
                    y=weather_percentages.values
                )
            ],

            'layout': {
                'title': 'Proportion of food issues caused by different weather types',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Weather Issue"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()