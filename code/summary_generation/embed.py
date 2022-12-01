'''
Applies pre-trained skip-thought encoder to captions and summaries.
'''
import pickle
import numpy as np

from skipthought_ryankiros import load_model, Encoder


def process_captions(filepath):
    '''
    Converts each caption from our image captioning train dataset to a list of strings.

    Output: 
    - List of string
    '''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    captions_idx = data['train_captions']
    idx2word = data['idx2word']

    # preprocess captions
    caption_list = []
    
    for caption in captions_idx:
        caption_list.append(" ".join([idx2word[i] for i in caption]))

    return caption_list


def process_summaries(filepath):
    '''
    Converts each summary from our book/movie summary train dataset to a list of strings.

    Output: 
    - List of string
    '''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    summaries_idx = data['train_summaries']
    idx2word = data['idx2word']

    # preprocess summaries
    summary_list = []

    for summary in summaries_idx:
        summary_list.append(" ".join([idx2word[i] for i in summary]))

    return summary_list


def generate_skipthoughts(text_list, encoder, filepath):
    '''
    Encodes list of text to skipthought vectors and pickles.

    Input:
    - List of strings
    - Encoder
    - Output filepath
    '''
    vectors = encoder.encode(text_list)
    mean_vec = np.mean(vectors, axis=0)

    skipthoughts = {
        'skipthoughts': vectors,
        'style': mean_vec
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