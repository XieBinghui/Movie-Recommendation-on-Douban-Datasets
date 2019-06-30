from data_small import *
from train import *
from recommendation import *
from forward import *

entity,relation,item_list,rela_h,rela_r,rela_t = load_data_kg()
print(len(entity))
print(len(relation))
print(len(rela_h))
#usr,rate = load_data_rate(item_list,entity)
#print(len(rate))
#kg = construct_kg(rela_h,rela_r,rela_t)
#train_data,valid_data,test_data,usr_history_dict = load_data(rate)
#ripple = load_usr_ripple(catch,hops,usr_history_dict,kg)
#rec = recommendation(2,ripple,3,entity,relation,usr_history_dict)
#print(rec)
#train(ripple,train_data,valid_data,test_data,entity,relation)
#sess = tf.InteractiveSession()
#when extract model have to identity which value to be assigned
#saver = tf.train.Saver()
#saver.restore(sess,"/home/bigdatalab27/Downloads/Ripple/model/model.ckpt-45782")
#print("good")
#print(evaluation(sess,test_data,ripple,BATCH_SIZE,len(entity),len(relation)))
#print(len(rate))
