# coding: utf-8

import pymysql
import numpy as np

class dataAPI(object):

    def  __init__(self):
        pass

    def fetchRatings(self, fromDB=True):
        '''
        fromDB: fetch data from database if fromDB is True, 
                else fetch data from local file 'Ratings'.
        '''

        if fromDB:
            # 从数据库中获取信息并处理成目标格式：[[user_id, movie_id, rating]...]
            conn=pymysql.connect(user='root', password='140014.Cjy', db='douban', autocommit=True)
            with conn.cursor() as cur:
                # 获取电影在数据库中id和豆瓣id的对应信息
                num=cur.execute(r'''
                    select id, number from movie_original
                    ''')
                print(num, '(id, movie_id) pairs got.')
                rawMovieId=cur.fetchall()
                
                # 获取用户对电影的评分信息
                num=cur.execute(r''' 
                    select user_id, rates from user_original where rates!="{}"
                    ''')
                print(num, 'items affected.')
                rawUserMovie=cur.fetchall()

                # 获取总矩阵大小
                num_usr=cur.execute(r'''
                    select user_id from user_original
                    ''')
                num_mv=cur.execute(r'''
                    select number from movie_original
                    ''')

            # 从豆瓣id映射到数据库中编号的字典
            id_2_movieid=dict(rawMovieId)

            rawData=[(item[0], eval(item[1])) for item in rawUserMovie]
            # 预分配空间给目标数组
            Ratings=np.empty_like([[0,0,0] for i in range(num**2)])
            i=0
            for item in rawData:
                for key, value in item[1].items():
                    if value and id_2_movieid.get(key):
                        Ratings[i]=[item[0], id_2_movieid.get(key), value]
                        i+=1

            Ratings=Ratings[:i]
            np.save('Ratings.npy', Ratings)
            UVsize=np.asarray([num_usr, num_mv])
            np.save('Size_User_Movie.npy', np.array([num_usr, num_mv]))
        else:
            # 从本地文件Ratings中读取
            Ratings=np.load('Ratings.npy')
            UVsize=np.load('Size_User_Movie.npy')

        return Ratings, UVsize

    def splitSets(self, percentage=0.2):
        '''
        Split Ratings into trainingSet_ratings and testingSet_ratings,
        where arg 'percentage' decides the percentage testingSet will take up.
        Then store them into local files 'trainingSet.npy' and 'testingSet.npy'.
        '''
        if not 1>=percentage>=0:
            print('During splitting: Percentage not valid!')
            return (0, 0)
        try:
            Ratings=np.load('Ratings.npy')
            n=len(Ratings)
            teNum=round(n*percentage)
            trNum=n-teNum
            print('Total:', n, 'Training:', trNum, 'Testing:', teNum)
            np.random.shuffle(Ratings)
            np.save('trainingSet.npy', Ratings[: trNum])
            np.save('testingSet.npy', Ratings[trNum:])
            return (trNum, teNum)
        except:
            print('During splitting: An unexpected error occurred!')
            return (0, 0)

    def readTrainingSet(self):
        try:
            data=np.load('trainingSet.npy')
            return data
        except:
            print('During reading file "trainingSet.npy", an error occurred.')
            return None
    
    def readTestingSet(self):
        try:
            data=np.load('testingSet.npy')
            return data
        except:
            print('During reading file "testingSet.npy", an error occurred.')
            return None

    def generateRemark(self, size=(1,1), ratings=[[0,0,0]], fromFile=False, save2File=False):
        '''
        Generate R matrix at size=size, using data from ratings.
        Ratings can be like data from readTestingSet().
        '''
        if fromFile:
            R=np.load('R.npy')
        else:
            R=np.zeros(size, dtype=int)
            for item in ratings:
                R[item[0]-1, item[1]-1]=item[2]
        if save2File:
            np.save('R.npy', R)

        return R



if __name__=='__main__':

    api=dataAPI()
    rating, uvsize=api.fetchRatings(fromDB=True)
    print(uvsize, rating)

    # trSet=api.readTrainingSet()
    # print(trSet, trSet.shape)
    # teSet=api.readTestingSet()
    # print(teSet, teSet.shape)

    api.splitSets()
    R=api.generateRemark(size=uvsize, ratings=rating, fromFile=False)
    print(R[64, 16042])

