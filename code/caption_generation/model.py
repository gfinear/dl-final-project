import numpy as np
import tensorflow as tf

class ImageCaptionModel(tf.keras.Model):

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    @tf.function
    def call(self, encoded_images, captions):
        return self.decoder(encoded_images, captions)  

    def get_config(self):
        return {"decoder": self.decoder}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        shuffle = tf.random.shuffle(range(len(train_captions)))
        train_captions_shuffle = tf.gather(train_captions, shuffle, axis = 0)
        train_image_features_shuffle = tf.gather(train_image_features, shuffle, axis = 0)

        num_batches = int(len(train_captions) / batch_size)
        avg_loss = 0
        avg_acc = 0
        avg_prp = 0 
        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_image_features = train_image_features_shuffle[start:end, :]
            decoder_input = train_captions_shuffle[start:end, :-1]
            decoder_labels = train_captions_shuffle[start:end, 1:]

            with tf.GradientTape() as tape:
                probs = self(batch_image_features, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)

     
        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc


def accuracy_function(prbs, labels, mask):
    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss