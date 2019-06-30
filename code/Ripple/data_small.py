import csv
import random
import numpy as np
import collections
from gl import *
def load_data_kg():
	"""处理电影的csv文件，并产生知识图谱中所需要的三元组"""
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	#创建kg中的实体字典
	entity = dict() #需要返回的值
	item_list = [] #需要返回的值
	count = 0
	for item in reader:
		if item[4] not in entity.keys():
			entity[item[4]] = count #电影id
			item_list.append(count) #电影需要进行单独的记录
			count = count + 1
	movieFile.close()
	relation = dict() #需要返回的值
	relation_count = 0
	#知识图谱中的三元组
	rela_h = []
	rela_r = []
	rela_t = []
	
	#根据电影的特征 选取关系 构建图谱
	#对评分信息进行处理
	
	relation['movie_movie_rate'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[1] == None: continue
		if item[1] not in entity.keys():
			entity[item[1]] = count #评分
			count = count + 1
		rela_h.append(entity[item[4]])
		rela_r.append(relation['movie_movie_rate'])
		rela_t.append(entity[item[1]])
	
	movieFile.close()
	
	#对导演信息进行处理
	
	relation['movie_movie_director'] = relation_count
	relation_count = relation_count + 1
	relation['movie_director_movie'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[5] == None: continue
		director = item[5]
		if director.strip() not in entity.keys():
			entity[director.strip()] = count #导演
			count = count + 1
		rela_h.append(entity[item[4]])
		rela_r.append(relation['movie_movie_director'])
		rela_t.append(entity[director.strip()])
		rela_h.append(entity[director.strip()])
		rela_r.append(relation['movie_director_movie'])
		rela_t.append(entity[item[4]])
	
	movieFile.close()
	
	#对演员信息进行处理
	
	relation['movie_movie_actor'] = relation_count
	relation_count = relation_count + 1
	relation['movie_actor_movie'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[7] == None: continue
		#选取主演
		for actor in item[7].split(",")[0:3]:
			if actor.strip() not in entity.keys():
				entity[actor.strip()] = count #演员
				count = count + 1
			rela_h.append(entity[item[4]])
			rela_r.append(relation['movie_movie_actor'])
			rela_t.append(entity[actor.strip()])
			rela_h.append(entity[actor.strip()])
			rela_r.append(relation['movie_actor_movie'])
			rela_t.append(entity[item[4]])
			
	movieFile.close()
	
	#对上映时间信息进行处理
	
	relation['movie_movie_year'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[6] == None: continue
		if item[6] not in entity.keys():
			entity[item[6]] = count #上映时间
			count = count + 1
		rela_h.append(entity[item[4]])
		rela_r.append(relation['movie_movie_year'])
		rela_t.append(entity[item[6]])
		
	
	movieFile.close()
	
	#对电影时长信息进行处理，小数据集中无
   	
	#对电影类型信息进行处理
	"""
	relation['movie_movie_type'] = relation_count
	relation_count = relation_count + 1
	relation['movie_type_movie'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[8] == None: continue
		for type_ in item[8].split(","):
			if type_.strip() not in entity.keys():
				entity[type_.strip()] = count #类型
				count = count + 1
			rela_h.append(entity[item[4]])
			rela_r.append(relation['movie_movie_type'])
			rela_t.append(entity[type_.strip()])
			rela_h.append(entity[type_.strip()])
			rela_r.append(relation['movie_type_movie'])
			rela_t.append(entity[item[4]])
			
	movieFile.close()
	"""
	#对上映地区信息进行处理
	relation['movie_movie_country'] = relation_count
	relation_count = relation_count + 1
	movieFile = open("movie_small.csv","r",encoding='utf-8')
	reader = csv.reader(movieFile)
	for item in reader:
		if item[9] == None: continue
		if item[9] not in entity.keys():
			entity[item[9]] = count #电影时长
			count = count + 1
		rela_h.append(entity[item[4]])
		rela_r.append(relation['movie_movie_country'])
		rela_t.append(entity[item[9]])

	movieFile.close()
	
	return entity,relation,item_list,rela_h,rela_r,rela_t
	
def load_data_rate(item_list,entity):
	"""本函数返回用户即评分记录"""
	#对于用户观看过但是没有评分的数据以一定的较大的概率将其设置为正样本
	#因为用户的观影记录可能过大，所以用户所看的电影都限制到90
	#因为时间递减，所以捕获的是最近的兴趣
	csv.field_size_limit(500 * 1024 * 1024) #设置否则会超过csv可读取行的大小
	usrFile = open("user_small.csv","r",encoding='utf-8')
	reader = csv.reader(usrFile)
	usr_count = 0
	usr = dict()
	rate = []
	user_pos_ratings = dict() #根据历史判定正样本还是负样本
	user_neg_ratings = dict()
	p = 0.8 #以0.8的概率将未评分的数据加入到正样本
	for u in reader:
		if u[0] not in usr:
			usr[u[0]] = usr_count
			usr_count = usr_count + 1
			user_pos_ratings[usr[u[0]]] = set()
			user_neg_ratings[usr[u[0]]] = set()
		for pair in u[2][1:len(u[2])-1].split(","):
			try:
				it,score =  pair.split(":")[0].strip(),pair.split(":")[1].strip()
				it = it[2:len(it)-1]
				print(it)
				score = score[2:len(score)-1]
				if it in entity.keys():
					if score == 'None':
						if random.random() <= p:
							user_pos_ratings[usr[u[0]]].add(entity[it])
							rate.append((usr[u[0]],1,entity[it]))
						else:user_neg_ratings[usr[u[0]]].add(entity[it])
					elif int(score) >= 4:
						user_pos_ratings[usr[u[0]]].add(entity[it])
						rate.append((usr[u[0]],1,entity[it]))
					else :
						user_neg_ratings[usr[u[0]]].add(entity[it])
			except:
				continue
		unwatch_set = set(item_list) - user_pos_ratings[usr[u[0]]] -  user_neg_ratings[usr[u[0]]]
		for it in np.random.choice(list(unwatch_set), size=len( user_pos_ratings[usr[u[0]]] ), replace=True):
			rate.append((usr[u[0]],0,it))
	return usr,rate

def load_data(rate):
	"""分割数据集用来train valid test，并且返回用户历史记录，用于生成ripple"""
	#分割数据集
	valid_ratio = 0.1
	test_ratio = 0.2
	n_rate = len(rate)

	#获取三个数据集的下标
	valid_indices = np.random.choice(n_rate,size=int(valid_ratio*n_rate),replace = False)
	left = set(range(n_rate)) - set(valid_indices)
	test_indices = np.random.choice(list(left),size=int(test_ratio*n_rate),replace = False)
	train_indices = list(left - set(test_indices))

	usr_history_dict = dict()
	for k in train_indices:
		u = rate[k][0]
		rating = rate[k][1]
		i = rate[k][2]
		if rating == 1:
			if u not in usr_history_dict:
				usr_history_dict[u] = []
			usr_history_dict[u].append(i)
	#因为要有hops 即需要对train中存在的用户才可以测试	
	train_indices = [i for i in train_indices if rate[i][0] in usr_history_dict]
	valid_indices = [i for i in valid_indices if rate[i][0] in usr_history_dict]
	test_indices = [i for i in test_indices if rate[i][0] in usr_history_dict]
	rate = np.array(rate)
	train_data = rate[train_indices]
	valid_data = rate[valid_indices]
	test_data = rate[test_indices]
	print(len(train_data))
	print(len(valid_data))
	print(len(test_data))
	
	return train_data,valid_data,test_data,usr_history_dict

def construct_kg(rela_h,rela_r,rela_t):
	kg = collections.defaultdict(list)
	count = len(rela_h)
	for i in range(count):
		kg[rela_h[i]].append((rela_r[i],rela_t[i]))	
	return kg
	

def load_usr_ripple(catch,hops,usr_history_dict,kg):
	#获取每一个user的ripple_set 便于训练时的使用
	#设置一个值，每个user在每一次hop选取的KG中的三元组相同，便于batch_size
	ripple = collections.defaultdict(list)
	for usr in usr_history_dict:
		tails = usr_history_dict[usr]
		for turn in range(hops):
			h = []
			r = []
			t = []
			for i in tails:
				#尾做头 传递
				for (rela,tail) in kg[i]:
					h.append(i)
					r.append(rela)
					t.append(tail)
			if len(h) == 0:
				#若没有进行传递，则取上一轮的hop
				ripple[usr].append((heads,relas,tails))
			else:
				indices = np.random.choice(len(h),catch,replace=(len(h)<catch))
				#若小于需要的三元组多少，则需要replace
				h = [h[i] for i in indices]
				r = [r[i] for i in indices]
				t = [t[i] for i in indices]
				ripple[usr].append((h,r,t))
				tails = t
				heads = h
				relas = r	
	return ripple
