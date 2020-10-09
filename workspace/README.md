# DS-Project2

Udacity Data Science Nanodegree Project 2 - Disaster Response Pipeline

### 1. Project Summary
The purpose of this project is to take messages extracted from a web based source (e.g. Twitter) and categorise
them using the content of the message in order to deploy the appropriate response. There are 36 categories that a
message can fit into.

### 2. Libraries Required to run:
In order to run the project you will need to install the following libraries:

* pandas
* nltk
* scikit-learn
* json
* pickle
* sqlalchemy
* flask
* plotly

### 3. Running Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 4. Files in the Repository
Model data - the messages used to train the model are in data/disaster_messages.csv. The corresponding categories
for the training data are in data/disaster_categories.csv

ETL script - the data is extracted, cleaned and loaded into a SQL database (data/DisasterResponse.db) by the script in 
data/process_data.py

Model script - the script to train the model (models/classifier.pkl) is in models/train_classifier.py

Web App - the script to launch the flask web app is in app/run.py 
