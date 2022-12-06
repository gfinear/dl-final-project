'''
Applies pre-trained skip-thought encoder to captions and summaries.
'''
import pickle
import numpy as np

from skipthought_ryankiros.skipthoughts import load_model, Encoder


def process_captions(filepath):
    '''
    Converts each caption from our image captioning train dataset to a list of strings.

    Output: 
    - List of string
    '''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    train_captions_idx = data['train_captions']
    test_captions_idx = data['test_captions']
    idx2word = data['idx2word']

    # preprocess captions
    train_captions = []
    test_captions = []
    
    for caption in train_captions_idx:
        train_captions.append(" ".join([idx2word[i] for i in caption]))

    for caption in test_captions_idx:
        test_captions.append(" ".join([idx2word[i] for i in caption]))

    return train_captions, test_captions


def process_summaries(filepath):
    '''
    Converts each summary from our book/movie summary train dataset to a list of strings.

    Output: 
    - List of string
    '''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    train_summaries_idx = data['train_summaries']
    test_summaries_idx = data['test_summaries']
    idx2word = data['idx2word']

    # preprocess summaries
    train_summaries = []
    test_summaries = []

    for summary in train_summaries_idx:
        train_summaries.append(" ".join([idx2word[i] for i in summary]))

    for summary in test_summaries_idx:
        test_summaries.append(" ".join([idx2word[i] for i in summary]))

    return train_summaries, test_summaries


def generate_skipthoughts(text_list, encoder, filepath):
    '''
    Encodes list of text to skipthought vectors and pickles.

    Input:
    - List of strings
    - Encoder
    - Output filepath
    '''
    train_list, test_list = text_list

    train_vectors = encoder.encode(train_list)
    test_vectors = encoder.encode(test_list)

    train_mean = np.mean(train_vectors, axis=0)
    test_mean = np.mean(test_vectors, axis=0)

    skipthoughts = {
        'train_skipthoughts': train_vectors,
        'test_skipthoughts': test_vectors,
        'train_style': train_mean,
        'test_style': test_mean
    }

    with open(filepath, 'wb') as f:
        pickle.dump(skipthoughts, f)

    print(f'Skipthoughts generated at {filepath}')


def main():
    # creates skipthought encoder model from pretrained model
    model = load_model()
    encoder = Encoder(model)

    data_filepath = "../../data/"
    captions = process_captions(data_filepath + "caption_data.p")
    summaries = process_summaries(data_filepath + "summary_data.p")

    output_filepath = "skipthought_embeddings/"
    caption_pickle = output_filepath + "caption_skipthoughts.p"
    summary_pickle = output_filepath + "summary_skipthoughts.p"

    generate_skipthoughts(captions, encoder, caption_pickle)
    generate_skipthoughts(summaries, encoder, summary_pickle)


if __name__ == '__main__':
    main()