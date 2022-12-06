import os
import pickle
import tensorflow as tf

from decoder import Decoder


def perplexity(truth, predicted):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    return tf.exp(scce(truth, predicted))

def main():
    cur = os.path.dirname(__file__)
    summary_path = os.path.join(cur, '../../data/summary_data.p')
    st_path = os.path.join(cur, 'skipthought_embeddings/summary_skipthoughts.p')

    with open(summary_path, 'rb') as sp, open(st_path, 'rb') as stp:
        summary_data = pickle.load(sp)
        skipthought_data = pickle.load(stp)

    vocab_size = len(summary_data['idx2word'])
    hidden_size = 128

    decoder = Decoder(vocab_size, hidden_size)

    decoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[perplexity]
    )

    # TODO: ENCODE TEST SUMMARIES AND FIX TO USE TEST

    train = summary_data['train_summaries'][:20000]
    test = summary_data['train_summaries'][20000:]

    train_context = skipthought_data['skipthoughts'][:20000]
    test_context = skipthought_data['skipthoughts'][20000:]

    decoder((train[:5], train_context[:5]))
    decoder.summary()

    decoder.fit(
        (train[:, :-1], train_context), train[:, 1:],
        epochs=10,
        batch_size=128,
        validation_data=((test[:, :-1], test_context), test[:, 1:])
    )


if __name__ == '__main__':
    main()