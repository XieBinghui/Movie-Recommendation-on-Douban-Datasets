import tensorflow as tf
import numpy as np
import random
import os
import math

from itertools import combinations
from collections import deque, Counter, OrderedDict


class BatchGenerator(object):
    # generate the batch
    def __init__(self, batch_size, data):
        self.batch_size =batch_size
        self.data = data
        self.index = 0
        self.batch = deque([]) # to store processed data
        self.finish = False # the flag to determine if we need go into next epoch

    def next(self):
        if self.finish is True:
            return 'no data'

        while len(self.batch) < self.batch_size:
            if self.data:
                movies = self.data[self.index]
                self.batch.extend(combinations(movies, 2)) # combine the movies in the same like_list or dislike_list
                self.index += 1

                if self.index == len(self.data):
                    self.finish = True
                    self.index = 0

        batch_data = [self.batch.popleft() for _ in range(self.batch_size)] # get the batch
        batch_data = np.array([[i[0], i[1]] for i in batch_data]) # split the data and label
        batch = batch_data[:, 0] # batch is a list of movies, (bacth_size, 1)

        # lables is also a list of movies which are in the same like_list or dislike_list, (batch_size, 1)
        labels = batch_data[:, 1]

        return batch, labels

    def shuffle_data(self):
        #index = [i for i in range(len(self.data))]
        #random.shuffle(index)
        self.data = np.random.shuffle(self.data)

    def rollback(self):
        self.index = 0
        self.finish = False
        # self.shuffle_data()


class Item2Vec(object):

    def __init__(self, session, data,
                    embed_dim=100,
                    negatives = 20,
                    learning_rate=0.001,
                    batch_size=250,
                    save_path='./model'):

        self.embed_dim = embed_dim
        self.negatives = negatives
        self.lr = learning_rate
        self.batch_size = batch_size
        self.save_path = save_path
        self.step = 0
        self.data = data
        self.movies_dict = self.get_movies()
        self.movies_list = list(self.movies_dict.keys())  # get words
        print(self.movies_list)
        self.movies_counts = list(self.movies_dict.values())
        # self.movies_counts = list(self.movies_counts/np.linalg.norm(self.movies_counts))
        self.movies_size = len(self.movies_list)
        self.data_id = self.map_to_ix() # data_id represent movies and range is [0:movies_size]
        self.batch_data = BatchGenerator(self.batch_size, self.data_id)
        self.step = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.sess = session
        self.build_graph()

    def map_to_ix(self):
        movies_to_ix = dict(zip(self.movies_list, range(len(self.movies_list))))
        self.movies_to_ix = movies_to_ix
        data_id = []
        for d in self.data:
            dl = []
            for x in d:
                if x in movies_to_ix.keys():
                    dl.append(movies_to_ix[x])
            data_id.append(dl)
        return data_id

    def build_graph(self):
        self.batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        positive_logits, negtive_logits = self.forward(self.batch, self.labels)
        self.loss_op = self.nce_loss(positive_logits, negtive_logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        #self.summary = tf.summary.FileWriter(self.save_path, graph=tf.get_default_graph())
        #self.merged = tf.summary.merge_all()

    def forward(self, batch, labels):
        # the embedding matrix: [movies_size, embed_dim]
        embed = tf.Variable(tf.random_uniform([self.movies_size, self.embed_dim],
                                              -1.0, 1.0), name='word_embedding')
        self.embed = embed
        # Output layer weights: [movies_size, embed_dim]
        self.soft_weight = tf.Variable(tf.zeros([self.movies_size, self.embed_dim]),
                                name="softmax_weights")
        # Output layer bias: [movies_size, embed_dim]
        self.soft_bias = tf.Variable(tf.ones([self.movies_size]), name='softmax_bias')
        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64),
                                   [self.batch_size, 1])

        # negative samples
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.negatives,
            unique=True,
            range_max=self.movies_size,
            distortion=0.75,
            unigrams=self.movies_counts)
        # sampled_ids = np.random.choice(self.movies_size, self.negatives, replace=False, p=self.movies_counts)

        # the embedding vectors for bacth
        batch_emb = tf.nn.embedding_lookup(embed, batch)
        # the output weights for bacth
        positive_w = tf.nn.embedding_lookup(self.soft_weight, labels)
        # the output bias for batch
        positive_b = tf.nn.embedding_lookup(self.soft_bias, labels)

        # weights for negative samples
        negtive_w = tf.nn.embedding_lookup(self.soft_weight, sampled_ids)
        # biases for negative samples
        negtive_b = tf.nn.embedding_lookup(self.soft_bias, sampled_ids)

        # the positive logits for positive samples
        positive_logits = tf.reduce_sum(tf.multiply(batch_emb, positive_w), 1) + positive_b
        # the negative logits for negative samples
        negtive_b = tf.reshape(negtive_b, [self.negatives])
        negtive_logits = tf.matmul(batch_emb,
                                   negtive_w,
                                   transpose_b=True) + negtive_b

        return positive_logits, negtive_logits

    def nce_loss(self, positive_logits, negative_logits):
        positive_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(positive_logits), logits=positive_logits)

        negative_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(negative_logits), logits=negative_logits)

        nce_loss = (tf.reduce_sum(positive_loss) +
                       tf.reduce_sum(negative_loss)) / self.batch_size

        return nce_loss

    def get_movies(self):
        movies = [x for lst in self.data for x in lst]
        counter = Counter(movies).most_common(len(movies))
        return OrderedDict(sorted(filter(lambda v: v[1] > 400, counter), reverse=True, key=lambda v: v[1]))

    def train(self):
        total_loss = 0
        while not self.batch_data.finish:
            batch, labels = self.batch_data.next()
            feed_dict = {self.batch: batch, self.labels: labels}
            #_, loss_batch, summary = self.sess.run([self.train_op, self.loss_op, self.merged],
            #                               feed_dict=feed_dict)
            _, loss_batch = self.sess.run([self.train_op, self.loss_op],
                                                  feed_dict=feed_dict)

            #self.summary.add_summary(summary, self.step)

            total_loss += loss_batch
            self.step += 1

        # avg_loss = total_loss / int(self.movies_size / self.batch_size)
        avg_loss = total_loss
        print("Cost:{:.4f}".format(avg_loss))
        self.batch_data.rollback()
        self.saver.save(self.sess, self.save_path, global_step=self.step)

    def eval(self, movie, N=5):
        ix = self.movies_to_ix[movie]
        for i, score in self.similar_items(ix, N):
            print(self.movies_list[i], score)
        print('-' * 10)
        print(self.embed)

    def similar_items(self, id, N=5):
        embed = self.embed.eval()
        norms = self.get_norms(embed)
        scores = embed.dot(embed[id]) / norms
        topN = np.argpartition(scores, -N)[-N:]
        return sorted(zip(topN, scores[topN] / norms[id]), key=lambda x: -x[1])

    def get_norms(self,embed):
        norms = np.linalg.norm(embed, axis=-1)
        norms[norms == 0] = 1e-10
        return norms


def Model(sess, data, embed_dim, negatives, learning_rate=1e-3, epoch=10, batch_size=250, save_path='./model'):
    model = Item2Vec(sess, data, embed_dim, negatives, learning_rate, batch_size, save_path)
    # print(len(data))
    for epoch in range(epoch):
        model.train() # Process one epoch

    # print('Finish {} epoch!'.format(epoch + 1))
    # print('-'*10)
    # model.eval(movie='25662329')
    return model
