import tensorflow as tf
from gl import *
import numpy as np

def forward(x,put_h,put_r,put_t,n_entity,n_relation):
	#前向传播的过程
	with tf.variable_scope('layer',reuse=tf.AUTO_REUSE):
		entity_embeddings = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,\
                                    shape=[n_entity, dim],initializer=tf.contrib.layers.xavier_initializer())
		relation_embeddings = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,\
                                    shape=[n_relation, dim,dim],initializer=tf.contrib.layers.xavier_initializer())
		transform_matrix = tf.get_variable(name="transform_matrix",dtype=tf.float64,shape=[dim,dim],initializer=tf.contrib.layers.xavier_initializer())
	#在矩阵中得到由index指向的embeddings
		item_embeddings = tf.nn.embedding_lookup(entity_embeddings,x)
		h_emb = []
		r_emb = []
		t_emb = []
		for hop in range(hops):
			h_emb.append(tf.nn.embedding_lookup(entity_embeddings, put_h[hop]))
			r_emb.append(tf.nn.embedding_lookup(relation_embeddings, put_r[hop]))
			t_emb.append(tf.nn.embedding_lookup(entity_embeddings, put_t[hop]))


	#每一轮的兴趣权重
		o_list = [] 
		for hop in range(hops):
			h_ = tf.expand_dims(h_emb[hop],axis=3)
			Rh = tf.squeeze(tf.matmul(r_emb[hop], h_), axis=3)
			v = tf.expand_dims(item_embeddings,axis=2)
			probs = tf.nn.softmax(tf.squeeze(tf.matmul(Rh, v), axis=2))
                #此次兴趣传播hop的相关概率o
			probs_ = tf.expand_dims(probs, axis=2)
		#没有用转置，使用了内积的形式，所以需要reduce_su
			o = tf.reduce_sum(t_emb[hop] * probs_, axis=1)
			o_list.append(o)
		#每次利用滑动平均
			item_embeddings = tf.matmul(item_embeddings + o,transform_matrix)
		usr_embedding = o_list[-1]
		for i in range(hops-1):
			usr_embedding  = usr_embedding + o_list[i]
		y_ = tf.reduce_sum(item_embeddings * usr_embedding, axis=1)
	#归一化
		y_predict = tf.sigmoid(tf.squeeze(y_))
		#one = tf.ones_like(y_predict)
		#zero = tf.zeros_like(y_predict)
		#y_predict = tf.where(y_predict<=0.5,x=zero,y=one)                
	#构建部分损失函数
	#由数据是否会过拟合判断是否需要使用kg loss 和 L2 loss
		kge_loss = 0
		l2_loss = 0.00000001
		kge_weight = 0.00000001
		l2_weight = 0
		for hop in range(hops):
			h_expanded = tf.expand_dims(h_emb[hop], axis=2)
			t_expanded = tf.expand_dims(t_emb[hop], axis=3)
			hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, r_emb[hop]), t_expanded))
			kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
		kge_loss = -kge_weight * kge_loss
		for hop in range(hops):
			l2_loss += tf.reduce_mean(tf.reduce_sum(h_emb[hop] * h_emb[hop]))
			l2_loss += tf.reduce_mean(tf.reduce_sum(t_emb[hop] * t_emb[hop]))
			l2_loss += tf.reduce_mean(tf.reduce_sum(r_emb[hop] * r_emb[hop]))
		l2_loss = l2_weight * l2_loss
		l2_loss = l2_loss + tf.nn.l2_loss(transform_matrix)                
	return y_predict,l2_loss+kge_loss
