import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from pathlib import Path
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
from keras.utils import custom_object_scope

class AttentionLayer(Layer):
  def __init__(self, **kwargs):
      super(AttentionLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    assert isinstance(input_shape, list)

    self.W_a = self.add_weight(name='W_a',
                                shape=(input_shape[0][2], input_shape[0][2]),
                                initializer='uniform',
                                trainable=True)
    self.U_a = self.add_weight(name='U_a',
                                shape=(input_shape[1][2], input_shape[0][2]),
                                initializer='uniform',
                                trainable=True)
    self.V_a = self.add_weight(name='V_a',
                                shape=(input_shape[0][2], 1),
                                initializer='uniform',
                                trainable=True)
    super(AttentionLayer, self).build(input_shape)

  def call(self, inputs):
    encoder_outputs, decoder_outputs = inputs  # encoder: [B, Tx, H], decoder: [B, Ty, H]

    # Linear projections
    W_a_dot_enc = tf.tensordot(encoder_outputs, self.W_a, axes=[[2], [0]])  # [B, Tx, H]
    U_a_dot_dec = tf.tensordot(decoder_outputs, self.U_a, axes=[[2], [0]])  # [B, Ty, H]

    # Broadcast add: expand dimensions
    WU_add = tf.nn.tanh(tf.expand_dims(U_a_dot_dec, 2) + tf.expand_dims(W_a_dot_enc, 1))  # [B, Ty, Tx, H]

    # Score → [B, Ty, Tx]
    score = tf.squeeze(tf.tensordot(WU_add, self.V_a, axes=[[3], [0]]), axis=-1)

    # Attention weights
    attention_weights = tf.nn.softmax(score, axis=-1)  # [B, Ty, Tx]

    # Context vector: [B, Ty, H]
    context_vector = tf.matmul(attention_weights, encoder_outputs)

    return context_vector, attention_weights



  def compute_output_shape(self, input_shape):
    return input_shape[1], input_shape[0]


base_dir = Path(__file__).resolve().parent.parent
print(f'base dir: {base_dir}')
# Load models
with custom_object_scope({'AttentionLayer': AttentionLayer}):
    decoder_model = load_model(base_dir.joinpath("models","decoder_model.h5"), compile=False)
    encoder_model = load_model(base_dir.joinpath("models","encoder_model.h5"), compile=False)

with open(base_dir.joinpath("models","target_word_index.pkl"), "rb") as f:
    target_word_index = pickle.load(f)

with open(base_dir.joinpath("models","reverse_target_word_index.pkl"), "rb") as f:
    reverse_target_word_index = pickle.load(f)

with open(base_dir.joinpath("models","x_tokenizer.pkl"), "rb") as f:
    x_tokenizer = pickle.load(f)

# Define constants (use same values from training)
max_len_text=80
max_len_summary=10

def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # Start token
    target_seq = np.zeros((1,1))
    target_seq[0,0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index.get(sampled_token_index, '')
        if(sampled_token != 'end'):
            decoded_sentence += ' ' + sampled_token

        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update target sequence
        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index

        # Update states
        e_h, e_c = h, c

    return decoded_sentence.strip()

