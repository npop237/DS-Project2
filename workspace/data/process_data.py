import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the Disaster response data from csv
    :param messages_filepath: filepath for the raw messages - 2nd argument in command line to run the file
    :param categories_filepath: filepath for the categories - 3rd argument in command line to run the file
    :return: merged dataframe of the messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')

    return df


def clean_data(df):
    """
    Cleans the data by splitting categories into individual columns ready for modelling
    :param df: merged dataframe of messages and categories
    :return: cleaned dataframe ready for modelling
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(subset=['message'])

    return df


def save_data(df, database_filename):
    """
    Creates a SQL database for the clean data
    :param df: cleaned messages data
    :param database_filename: 4th argument in the command line when running the file
    :return: nothing
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    engine.text_factory = str
    df.to_sql('Messages_Categories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()