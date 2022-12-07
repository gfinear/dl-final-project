'''
Decodes skip-thought vector into summaries.
'''
import os
import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size=None, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        if vocab_size is None:
            cur = os.path.dirname(__file__)
            path = os.path.join(cur, "summary_model")
            custom = {'perplexity': self.perplexity}
            self.decoder = tf.keras.models.load_model(path, custom_objects=custom)
        else:
            self.skipthought_embedding = tf.keras.layers.Dense(hidden_size)
            self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
            self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True)
            self.classifier = tf.keras.layers.Dense(vocab_size)

            self.perplexity_tracker = tf.keras.metrics.Mean(name='perplexity')

    def call(self, inputs):
        if self.vocab_size is None:
            return self.decoder(inputs)

        summaries, st_vectors = inputs
        st_embeddings = self.skipthought_embedding(st_vectors)
        summary_embeddings = self.embedding(summaries)
        outputs = self.gru(summary_embeddings, initial_state=st_embeddings)
        logits = self.classifier(outputs)
        return logits

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def batch_step(self, data, training):
        inputs, test = data

        with tf.GradientTape() as tape:
            result = self(inputs)
            loss = self.compiled_loss(test, result)

        if training:
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.perplexity_tracker.update_state(tf.exp(loss))

        return {m.name: m.result() for m in self.metrics}

    def perplexity(self, truth, predicted):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return tf.exp(scce(truth, predicted))

    # TODO: Add temperature function