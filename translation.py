import tensorflow as tf
import os
import io
import re
from sklearn.model_selection import train_test_split
from time import time
from tqdm import tqdm
from matplotlib import pyplot, ticker
import numpy as np


def preprocess_phrase(phrase):
    phrase = phrase.lower().strip()
    for contraction, expansion in [("don't", "do not"),
                                   ("i'm", "i am"),
                                   ("he's", "he is"),
                                   ("she's", "she is"),
                                   ("that's", "that is"),
                                   ("wouldn't", "would not"),
                                   ("has't", "has not"),
                                   ("didn't", "did not")]:
      phrase = re.sub(contraction, expansion, phrase)
    phrase = re.sub(r'([?.!,])', r' \1 ', phrase)
    phrase = re.sub(r'[" "]+', " ", phrase)
    phrase = re.sub(r'[^a-zA-Z0-9?.!,ɛɔŋ]+', ' ', phrase)
    phrase = phrase.strip()

    return f'<start> {phrase} <end>'


def create_dataset(file_path, num_examples):
    lines = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_phrase(phrase) for phrase in line.split('\t')] for line in lines[:num_examples]]

    print(f'{len(word_pairs)} phrases loaded')

    return zip(*word_pairs)


def tokenize(language):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(language)

    tensor = tokenizer.texts_to_sequences(language)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer


def load_dataset(file_path, num_examples=None):
    phrases_tokenized = []
    tokenizers = []
    for phrases in create_dataset(file_path, num_examples):
      tokens, tokenizer = tokenize(phrases)
      phrases_tokenized.append(tokens)
      tokenizers.append(tokenizer)

    return phrases_tokenized + tokenizers


def convert(language, phrase):
    for token in phrase:
        if token != 0:
            print(f'{token} ----> {language.index_word[token]}') 


def calculate_loss(truth, predictions):
  mask = tf.logical_not(tf.equal(truth, 0))
  cross_entropy = calculate_cross_entropy(truth, predictions, mask)

  loss = cross_entropy * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(tf.cast(mask, tf.float32))
  if not tf.math.is_nan(loss):
    return loss


# @tf.function
def train(phrase_input, phrase_output, state_encoder):
  loss = 0
  with tf.GradientTape() as tape:
    encoding, encoding_state = encoder(phrase_input, state_encoder)
    state_decoder = state_encoder
    decoder_input = tf.expand_dims([english.word_index['<start>']] * batch_size, 1)
    output_length = phrase_output.shape[1]
    for output_word_index in range(1, output_length):
      predictions, state_decoder, _ = decoder(decoder_input, state_decoder, encoding)
      output_token = phrase_output[:, output_word_index]
      word_loss = calculate_loss(output_token, predictions)
      if word_loss is None:
        continue
      loss += word_loss
      decoder_input = tf.expand_dims(phrase_output[:, output_word_index], 1)
    batch_loss = loss / output_length
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


def evaluate(sentence):
  sentence = preprocess_phrase(sentence)
  tokens = [english.word_index[word] for word in sentence.split(' ')]
  tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=max_length_ga, padding='post')
  tokens = tf.convert_to_tensor(tokens)

  result = ''
  state = [tf.zeros((1, num_units))]
  encoding, state_encoding = encoder(tokens, state)
  state_decoder = state_encoding
  decoder_inputs = tf.expand_dims([ga.word_index['<start>']], 0)

  attention_plot = []
  for index in range(max_length_ga):
    predictions, state_decoder, attention_weights = decoder(decoder_inputs, state_decoder, encoding)
    attention_weights = tf.reshape(attention_weights, [-1])
    attention_plot.append(attention_weights.numpy())

    tokens_predicted = tf.argmax(predictions[0]).numpy()
    result += ga.index_word[tokens_predicted] + ' '

    if ga.index_word[tokens_predicted] == '<end>':
      return result, sentence, attention_plot

    decoder_inputs = tf.expand_dims([tokens_predicted], 0)
  
  return result, sentence, attention_plot


def plot_attention(attention, sentence, sentence_predicted):
  sentence = sentence.split(' ')
  sentence_predicted = sentence_predicted.split(' ')
  attention = np.array(attention)[:len(sentence_predicted), :len(sentence)]

  figure = pyplot.figure(figsize=[10, 10])
  axes = figure.add_subplot(1, 1, 1)
  axes.matshow(attention, cmap='viridis')
  font = {'fontsize': 14}
  axes.set_xticklabels([''] + sentence, fontdict=font, rotation=90)
  axes.set_yticklabels([''] + sentence_predicted, fontdict=font, rotation=90)
  axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
  axes.yaxis.set_major_locator(ticker.MultipleLocator(1))

  pyplot.show()


def translate(sentence, plot=False):
  result, sentence, attention_plot = evaluate(sentence)

  print(f'input: {sentence}')
  print(f'translation: {result}')

  if plot:
    plot_attention(attention_plot, sentence, result),


class Encoder(tf.keras.Model):
  def __init__(self, vocabulary_size, embedding_size, num_encoding_units):
    super().__init__()
    self._num_encoding_units = num_encoding_units
    self._embed = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
    self._gru = tf.keras.layers.GRU(num_encoding_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

  def call(self, inputs, state):
    embededding = self._embed(inputs)
    
    return self._gru(embededding, initial_state=state)

  def initialize_state(self, batch_size):
    return tf.zeros((batch_size, self._num_encoding_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, num_units):
    super().__init__()
    self._w1 = tf.keras.layers.Dense(num_units)
    self._w2 = tf.keras.layers.Dense(num_units)
    self._v = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query = tf.expand_dims(query, 1)
    score = self._v(tf.nn.tanh(self._w1(query) + self._w2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocabulary_size, embedding_size, num_units):
    super().__init__()
    self._embed = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
    self._gru = tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self._dense = tf.keras.layers.Dense(vocabulary_size)
    self._attention = BahdanauAttention(num_units)

  def call(self, inputs, state, encoding):
    context_vector, attention_weights = self._attention(state, encoding)
    embedding = self._embed(inputs)
    embedding = tf.concat([tf.expand_dims(context_vector, 1), embedding], axis=-1)
    outputs, state = self._gru(embedding, initial_state=state)
    predictions = self._dense(tf.squeeze(outputs, 1))
    
    return predictions, state, attention_weights

# file_name = 'spa-eng.zip'
# file_path_zip = tf.keras.utils.get_file(file_name, origin=f'http://storage.googleapis.com/download.tensorflow.org/data/{file_name}', extract=True)
# file_path = os.path.join(file_path_zip, 'spa-eng/spa.txt')
file_path = 'C:\\Users\\niioa\\Google Drive\\ga\\ga-english.txt'

phrases_encoded_ga, phrases_encoded_english, ga, english = load_dataset(file_path)
max_length_ga = phrases_encoded_ga.shape[1]
max_length_english = phrases_encoded_english.shape[1]

phrases_english_train, phrases_english_validation, phrases_ga_train, phrases_ga_validation = train_test_split(phrases_encoded_english, phrases_encoded_ga, test_size=.2, random_state=123)

buffer_size = len(phrases_encoded_english)
batch_size = 64
num_steps_per_epoch = buffer_size // batch_size
embedding_size = 256
num_units = 1024
vocab_size_ga = len(ga.word_index) + 1
print(f'{vocab_size_ga} Ga words')
vocab_size_english = len(english.word_index) + 1
print(f'{vocab_size_english} English words')

dataset = tf.data.Dataset.from_tensor_slices((phrases_english_train, phrases_ga_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

encoder = Encoder(vocab_size_english, embedding_size, num_units)
decoder = Decoder(vocab_size_ga, embedding_size, num_units)

optimizer = tf.keras.optimizers.Adam()
calculate_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

log_directory = os.path.join('logs' ,'ga' ,'0')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint_prefix = os.path.join(log_directory, "ckpt")
manager = tf.train.CheckpointManager(checkpoint, log_directory, 3)

checkpoint.restore(tf.train.latest_checkpoint(log_directory))
num_log_steps = 100
num_save_epochs = 2
num_epochs = 50
for epoch in tqdm(range(num_epochs), 'epoch'):
  start = time()
  state_encoder = encoder.initialize_state(batch_size)
  loss_total = 0
  num_steps = 0
  for (batch_index, (phrases_english, phrases_ga)) in tqdm(enumerate(dataset), 'batch'):
    batch_loss = train(phrases_english, phrases_ga, state_encoder)
    loss_total += batch_loss

    if batch_index % num_log_steps == 0:
      print(f'epoch {epoch} batch index {batch_index} loss {batch_loss.numpy():.4f}')
      english_phrase = ' '.join([english.index_word[token.numpy()] for token in phrases_english[0] if token != 0][1:-1])
      translate(english_phrase)
      print(f'label: {" ".join([ga.index_word[token.numpy()] for token in phrases_ga[0] if token != 0][1:-1])}')
  
    num_steps = max(batch_index + 1, num_steps)

  if epoch % num_save_epochs == 0:
    # checkpoint.save(file_prefix=checkpoint_prefix)
    print('Saving checkpoint')
    manager.save()
  print(f'epoch {epoch} loss {loss_total / num_steps:.4f}')

manager.save

translate('how are you today', True)