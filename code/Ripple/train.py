#训练函数
#模型的参数
import tensorflow as tf
from forward import forward
import os
import numpy as np
from gl import *
information = open("save.txt",'w')
MODEL_NAME = "model.ckpt"
MODEL_SAVE_PATH = "/home/bigdatalab27/Downloads/Ripple/model_temp"
def train(ripple,train_data,valid_data,test_data,entity,relationship):
	n_entity = len(entity)
	n_relation = len(relationship)
	#确定输入数据
	x = tf.placeholder(dtype=tf.int32,shape=[None],name="items")
	y = tf.placeholder(dtype=tf.float64,shape=[None],name="labels")
	put_h = []
	put_r = []
	put_t = []
	for hop in range(hops):
		put_h.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
		put_r.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
		put_t.append(tf.placeholder(dtype=tf.int32,shape=[None,catch],name="h"+str(hop)))
	
	y_,loss = forward(x,put_h,put_r,put_t,n_entity,n_relation)
	loss = loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_))
	#持久化模型
	global_step = tf.Variable(0, trainable=False)
	train_step = tf.train.AdamOptimizer(LR) \
            .minimize(loss, global_step=global_step)
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for turn in range(TRAINING_STEPS):
                        np.random.shuffle(train_data)
                        start = 0
                        while start + BATCH_SIZE < len(train_data):
                                end = start + BATCH_SIZE
                                feed_dict = dict()
                                feed_dict[x] = train_data[start:end, 2]
                                feed_dict[y] = train_data[start:end, 1]
                                for i in range(hops):
                                        feed_dict[put_h[i]] = [ripple[user][i][0] for user in train_data[start:end, 0]]
                                        feed_dict[put_r[i]] = [ripple[user][i][1] for user in train_data[start:end, 0]]
                                        feed_dict[put_t[i]] = [ripple[user][i][2] for user in train_data[start:end, 0]]
                                _,loss_, step= sess.run([train_step, loss, global_step],feed_dict=feed_dict)
                                start = start + BATCH_SIZE
                        print("turn:%d,loss:%f"%(turn,loss_))
                        information.write("turn:%d,loss:%f"%(turn,loss_))
                        information.write("\n")
			# 每10轮保存一轮模型
                        if turn % 1 == 0:
                                train_acc,train_pre,train_recall,train_F1 = evaluation(sess, train_data, ripple, BATCH_SIZE,n_entity,n_relation)
                               	print("train_acc:%f,train_per:%f,train_recall:%f,train_F1:%f" % ( train_acc,train_pre,train_recall,train_F1))
                                #valid_acc,valid_pre,valid_recall,valid_F1 = evaluation(sess, valid_data, ripple, BATCH_SIZE,n_entity,n_relation)
                                #print("valid_acc:%f,valid_per:%f,valid_recall:%f,valid_F1:%f" % ( valid_acc,valid_pre,valid_recall,valid_F1)) 
                                #information.write("valid_acc:%f,valid_per:%f,valid_recall:%f,valid_F1:%f" % ( valid_acc,valid_pre,valid_recall,valid_F1)) 
                               	#information.write("\n") 
                               	test_acc,test_pre,test_recall,test_F1 = evaluation(sess, test_data, ripple, BATCH_SIZE,n_entity,n_relation)
                                print("test_acc:%f,test_per:%f,test_recall:%f,test_F1:%f" % ( test_acc,test_pre,test_recall,test_F1))
                                information.write("test_acc:%f,test_per:%f,test_recall:%f,test_F1:%f" % ( test_acc,test_pre,test_recall,test_F1))
                                information.write("\n")
                                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=turn)
	information.close()
def evaluation(sess,data,ripple,BATCH_SIZE,n_entity, n_relation):
    ylist = []
    y_list = []
    start = 0
    x = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
    y = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
    put_h = []
    put_r = []
    put_t = []
    for hop in range(hops):
        put_h.append(tf.placeholder(dtype=tf.int32, shape=[None, catch], name="h" + str(hop)))
        put_r.append(tf.placeholder(dtype=tf.int32, shape=[None, catch], name="h" + str(hop)))
        put_t.append(tf.placeholder(dtype=tf.int32, shape=[None, catch], name="h" + str(hop)))
    while start + BATCH_SIZE < len(data):
        end = start + BATCH_SIZE
        feed_dict = dict()
        feed_dict[x] = data[start:end, 2]
        feed_dict[y] = data[start:end, 1]
        for i in range(hops):
	        feed_dict[put_h[i]] = [ripple[user][i][0] for user in data[start:end, 0]]
	        feed_dict[put_r[i]] = [ripple[user][i][1] for user in data[start:end, 0]]
	        feed_dict[put_t[i]] = [ripple[user][i][2] for user in data[start:end, 0]]	
        y_, loss = forward(x, put_h, put_r, put_t, n_entity, n_relation)
        y1,y_1 = sess.run([y,y_],feed_dict=feed_dict)#在session 中的部分由前向函数得到y，所以可以在
        
        print(y1)
        print(y_1)
        start = start + BATCH_SIZE
        ynp = y1.tolist()
        y_np = y_1.tolist()
        for i in range(len(ynp)):
                ylist.append(ynp[i])
                y_list.append(y_np[i] >= 0.5)
    
    NF = 0
    NT = 0
    PT = 0
    PF = 0
    for i in range(len(ylist)):
        if ylist[i] == 1 and y_list[i] == 0:
               NF = NF + 1
        if ylist[i] == 0 and y_list[i] == 0:
               NT = NT + 1
        if ylist[i] == 1 and y_list[i] == 1:
               PT = PT + 1
        if ylist[i] == 0 and y_list[i] == 1:
               PF = PF + 1
    acc = (PT+NT) / (PT+NT+PF+NF)
    pre = PT / (PT + PF)
    recall = PT / (PT + NF)
    F1 = 2*pre*recall / (pre+recall)
    return acc,pre,recall,F1
