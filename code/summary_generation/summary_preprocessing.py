
import pickle
import random
import re
from PIL import Image
import numpy as np
import collections
import pandas as pd
import csv

def preprocess_summaries(summaries, window_size):
    for i, summary in enumerate(summaries):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

        # Convert the summary to lowercase, and then remove all special characters from it
        summary_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', summary.lower())
      
        # Split the summary into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_words = [word for word in summary_nopunct.split() if not ((len(word) == 1) and (word != "a"))]
      
        # Join those words into a string
        summary_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        summaries[i] = summary_new


def load_movies(data_folder):
    movie_summary_path = f'{data_folder}/moviesummaries/mpst_full_data.csv'
    movie_data = pd.read_csv(movie_summary_path, skipinitialspace=True)
    
    train_summaries = movie_data[movie_data['split'] == "train"]["plot_synopsis"].tolist()

    test_summaries = movie_data[movie_data['split'] == "test"]["plot_synopsis"].tolist()

    return train_summaries, test_summaries

def load_books(data_folder):
    book_summary_path = f'{data_folder}/booksummaries/booksummaries.txt'
    book_data = list(csv.reader(open(book_summary_path, 'r'), delimiter='\t'))
    summaries = [row[6] for row in book_data]

    random.seed(0)
    random.shuffle(summaries)
    test_summaries = summaries[:3300]
    train_summaries = summaries[3300:]

    return train_summaries, test_summaries

def process_summaries(train_summaries, test_summaries):
    #remove special charachters and other nessesary preprocessing
    window_size = 250
    preprocess_summaries(train_summaries, window_size)
    preprocess_summaries(test_summaries, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for summary in train_summaries:
        word_count.update(summary)

    def unk_summaries(summaries, minimum_frequency):
        for summary in summaries:
            for index, word in enumerate(summary):
                if word_count[word] <= minimum_frequency:
                    summary[index] = '<unk>'

    unk_summaries(train_summaries, 20)
    unk_summaries(test_summaries, 20)

    # pad summaries so they all have equal length
    def pad_summaries(summaries, window_size):
        for summary in summaries:
            summary += (window_size + 1 - len(summary)) * ['<pad>'] 
    
    pad_summaries(train_summaries, window_size)
    pad_summaries(test_summaries,  window_size)

    # assign unique ids to every work left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for summary in train_summaries:
        print(summary)
        for index, word in enumerate(summary):
            if word in word2idx:
                summary[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                summary[index] = vocab_size
                vocab_size += 1
    for summary in test_summaries:
        for index, word in enumerate(summary):
            if word in word2idx:
                summary[index] = word2idx[word] 
            else:
                summary[index] = word2idx['<unk>']

    return dict(
        train_summaries         = np.array(train_summaries),
        test_summaries          = np.array(test_summaries),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )

def load_data(data_folder):
    train_movies, test_movies = load_movies(data_folder)
    train_books, test_books = load_books(data_folder)

    train_data = train_movies + train_books
    test_data = test_movies + test_books

    return process_summaries(train_data, test_data)


def create_pickle(data_folder):
    with open(f'{data_folder}/summary_data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/summary_data.p!')


if __name__ == '__main__':
    # download book data here: https://www.cs.cmu.edu/~dbamman/booksummaries.html and movie data here: https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags 

    # make sure the book data is within 'data/booksummaries/booksummaries.txt' and the movie data is within 'data/moviesummaries/mpst_full_data.csv'

    # THIS MAY REQUIRE RENAMING FILES

    data_folder = '../../data'
    create_pickle(data_folder)