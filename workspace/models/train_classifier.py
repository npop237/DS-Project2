import sys
import pandas as pd
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads in the disaster messages to prepare for modelling
    :param database_filepath: 2nd argument in the command line when running the file
    :return: dataframes for the messages and the target categories along with a list of the category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='Messages_Categories', con=engine)
    # Drop null values from df
    df.dropna(inplace=True)
    # Split into x and Y
    X = df['message']
    Y = df.drop(columns=['id', 'original', 'genre', 'message'])
    categories = Y.columns

    return X, Y, categories


def tokenize(text):
    """
    Breaks up the words in our disaster message into lemmatized tokens for modelling
    :param text: input message
    :return: list of lemmatized words in the message
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Prepares the model using pipeline and cross validation to tune parameters
    :return: the model
    """
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__weights': ['uniform', 'distance']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Makes predictions based on our test split and evaluates against the test categories
    :param model: the cross validated pipeline model
    :param X_test: test messages
    :param Y_test: actual test categories
    :param category_names: the category names loaded from the load_data function
    :return: nothing - prints out the classification report
    """
    Y_pred = model.predict(X_test)
    for i, x in enumerate(category_names):
        print(x)
        print(classification_report(Y_test.values[i], Y_pred[i], target_names=["0", "1"]))


def save_model(model, model_filepath):
    """
    Saves the model into a pickle file to be read by the web app
    :param model: the cross validated pipeline model
    :param model_filepath: 3rd argument in the command line when running the file
    :return: nothing
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()