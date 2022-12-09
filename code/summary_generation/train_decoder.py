import pickle
import tensorflow as tf

from decoder import Decoder


def load_data():
    summary_path = '../../data/summary_data.p'
    st_path = 'skipthought_embeddings/summary_skipthoughts.p'

    with open(summary_path, 'rb') as sp, open(st_path, 'rb') as stp:
        summary_data = pickle.load(sp)
        skipthought_data = pickle.load(stp)

    return summary_data, skipthought_data


def main():
    summary_data, skipthought_data = load_data()
    vocab_size = len(summary_data['idx2word'])

    decoder = Decoder(vocab_size=vocab_size)

    decoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy()
    )

    train = summary_data['train_summaries']
    test = summary_data['test_summaries']

    train_context = skipthought_data['train_skipthoughts']
    test_context = skipthought_data['test_skipthoughts']

    decoder((train[:5], train_context[:5]))
    decoder.summary()

    decoder.fit(
        (train[:, :-1], train_context), train[:, 1:],
        epochs=2,
        batch_size=128,
        validation_data=((test[:, :-1], test_context), test[:, 1:])
    )

    decoder.save('summary_model')


if __name__ == '__main__':
    main()