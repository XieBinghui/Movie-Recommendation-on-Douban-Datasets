#构建推荐系统
import tensorflow as tf
from forward import forward
from gl import *
model_path = 'model_small_all/model.ckpt-19'
def recommendation(usr,ripple,N,entity,relationship,usr_history_dict):
	n_entity = len(entity)
	n_relation = len(relationship)
	#确定输入数据
	put_h = []
	put_r = []
	put_t = []
	x = tf.placeholder(dtype=tf.int32,shape=[None],name="items")
	for hop in range(hops):
		put_h.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
		put_r.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
		put_t.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
	
	y_,_ = forward(x,put_h,put_r,put_t,n_entity,n_relation)
	recommendation = dict()
	#对每个物品判断推荐的概率
	for item in range(n_entity):
		emm = [item]
		feed_dict = dict()
		feed_dict[x] = emm
		for i in range(hops):
			feed_dict[put_h[i]] = [ripple[usr][i][0]]
			feed_dict[put_r[i]] = [ripple[usr][i][1]]
			feed_dict[put_t[i]] = [ripple[usr][i][2]]
	
		sess = tf.InteractiveSession()

		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		y = sess.run([y_], feed_dict=feed_dict)
		#不推荐已经观看的电影
		if item not in usr_history_dict:
			if item not in recommendation.keys():
				recommendation[item] = y
	
	result = sorted(recommendation.items(),key = lambda item:item[1],reverse=True)
	
	return result.keys[0:N]

	


