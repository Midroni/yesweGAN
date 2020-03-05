# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""IMDB data loader and helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Dependency imports
import numpy as np

import tensorflow as tf

"""this is bad but im going to do it anyway"""
stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('prefix_label', True, 'Vocabulary file.')

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

EOS_INDEX = 88892


def _read_words(filename, use_prefix=False):
  all_words = []
  sequence_example = tf.train.SequenceExample()
  for r in tf.python_io.tf_record_iterator(filename):
    sequence_example.ParseFromString(r)

    if FLAGS.prefix_label and use_prefix:
      label = sequence_example.context.feature['class'].int64_list.value[0]
      review_words = [EOS_INDEX + 1 + label]
    else:
      review_words = []
    review_words.extend([
        f.int64_list.value[0]
        for f in sequence_example.feature_lists.feature_list['token_id'].feature
    ])
    all_words.append(review_words)
  return all_words


def build_vocab(vocab_file):
  word_to_id = {}

  with tf.gfile.GFile(vocab_file, 'r') as f:
    index = 0
    for word in f:
      word_to_id[word.strip()] = index
      index += 1
    word_to_id['<eos>'] = EOS_INDEX

  return word_to_id

def build_stopword_dict(word_to_id):
    stop_words_id = []
    for word in stop_words:
        if word in word_to_id:
            stop_words_id.append(word_to_id[word])
    print(stop_words_id)
    return stop_words_id


def imdb_raw_data(data_path=None):
  """Load IMDB raw data from data directory "data_path".
  Reads IMDB tf record files containing integer ids,
  and performs mini-batching of the inputs.
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data)
    where each of the data objects can be passed to IMDBIterator.
  """

  train_path = os.path.join(data_path, 'train_lm.tfrecords')
  valid_path = os.path.join(data_path, 'test_lm.tfrecords')

  train_data = _read_words(train_path)
  valid_data = _read_words(valid_path)
  return train_data, valid_data


def imdb_iterator(raw_data, batch_size, num_steps, epoch_size_override=None):
  """Iterate on the raw IMDB data.

  This generates batch_size pointers into the raw IMDB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from imdb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one. The third is a set of weights with 1 indicating a word was
    present and 0 not.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  del epoch_size_override
  data_len = len(raw_data)
  num_batches = data_len // batch_size - 1

  for batch in range(num_batches):
    x = np.zeros([batch_size, num_steps], dtype=np.int32)
    y = np.zeros([batch_size, num_steps], dtype=np.int32)
    w = np.zeros([batch_size, num_steps], dtype=np.float)

    for i in range(batch_size):
      data_index = batch * batch_size + i
      example = raw_data[data_index]

      if len(example) > num_steps:
        final_x = example[:num_steps]
        final_y = example[1:(num_steps + 1)]
        w[i] = 1

      else:
        to_fill_in = num_steps - len(example)
        final_x = example + [EOS_INDEX] * to_fill_in
        final_y = final_x[1:] + [EOS_INDEX]
        w[i] = [1] * len(example) + [0] * to_fill_in

      x[i] = final_x
      y[i] = final_y

    yield (x, y, w)


def imdb_iterator_custom(raw_data, batch_size, num_steps, stop_words_id, epoch_size_override=None):
  """Iterate on the raw IMDB data.

  This generates batch_size pointers into the raw IMDB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from imdb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one. The third is a set of weights with 1 indicating a word was
    present and 0 not.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  del epoch_size_override
  data_len = len(raw_data)
  num_batches = data_len // batch_size - 1

  for batch in range(num_batches):
    x = np.zeros([batch_size, num_steps], dtype=np.int32)
    y = np.zeros([batch_size, num_steps], dtype=np.int32)
    w = np.zeros([batch_size, num_steps], dtype=np.float)
    p = np.zeros([batch_size, num_steps], dtype=np.bool_)


    for i in range(batch_size):
      data_index = batch * batch_size + i
      example = raw_data[data_index]

      if len(example) > num_steps:
        final_x = example[:num_steps]
        final_y = example[1:(num_steps + 1)]
        w[i] = 1

      else:
        to_fill_in = num_steps - len(example)
        final_x = example + [EOS_INDEX] * to_fill_in
        final_y = final_x[1:] + [EOS_INDEX]
        w[i] = [1] * len(example) + [0] * to_fill_in
      final_p = []
      for k, x_tmp in enumerate(final_x):
          if x_tmp in stop_words_id:
            final_p[k] = False
            print('stop word:')
            print(x_tmp)
          else:
            final_p[k] = True
            print('not stop word:')
            print(x_tmp)

      x[i] = final_x
      y[i] = final_y
      p[i] = final_p

    yield (x, y, w, p)
