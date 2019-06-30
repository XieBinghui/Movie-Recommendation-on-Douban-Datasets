# coding: utf-8

from douban_spider.douban_tools import *
from multiprocessing import Queue
from random import randint
from time import sleep
import pymysql
import re



def single_User_Spider(user_queue, data_queue, maxNum, sess, conn, mplock=None):
    i=1
    q=1
    total=1
    maxQueueNum=maxNum//10

    while i<=maxNum:
        user_id=user_queue.get()
        q-=1
        with conn.cursor() as cur:
            # 在数据库中查找该用户信息
            in_db=cur.execute('select * from user where user_number=%s', (user_id,))
        if in_db:
            # 若在数据库中，则跳过此项
            print('User in db, skip one.')
            continue

        # 该用户不在数据库中，开始爬取数据
        try:
            user_data=get_user_info(user_id, sess)
            # 数据放入数据队列
            data_queue.put(user_data)
            if total<maxQueueNum:
                # 添加该用户关注的用户到用户队列中
                for item in user_data['following_id']:
                    user_queue.put(item)
                    total+=1
                    q+=1
            i+=1

        except TimeoutError:
            print('Timeout, skip one.')
            continue
        except TypeError as e:
            print(e) 
            if (e.__str__().startswith("'NoneType'")):
                print('403 detected. Process aborted.')
                break
            # 自动处理验证码
            if mplock:
                mplock.acquire()
                ok=verify(sess)
                mplock.release()
            else:
                ok=verify(sess)
            if not ok:
                break
        
        print('data queue:', i, 'user queue:', q)
        # data队列中每累计起3个便调用一次插入程序
        if i%3==0:
            insertData(data_queue, conn)
        
        # sleep(randint(4, 20))

    

def single_Movie_Spider(movie_queue, data_queue, maxNum, sess, conn, mplock=None):
    # 第一阶段，获取需要爬取的电影序号，并且放入电影队列（需要考虑的是是否需要爬取imdb评分，有好有坏：增加时延，减少403几率，同时降低了爬虫效率）
    # ，此阶段在主进程实现好了(调用本包中的getQueue），本函数仅仅使用参数movie_queue来利用队列中的电影序号
    i=1
    # 第二阶段，取出一个电影序号，在数据库中查找是否已经有了这个电影的相关信息，如果有了便跳过此次插入，如果没有：
    # 调用douban_tools中的get_movie_info爬取电影数据，注意返回值是一个字典，包含许多数据
    while i<=maxNum:
        movie_id=movie_queue.get()
        with conn.cursor() as cur:
            in_db=cur.execute('select * from movie where id=%s', (movie_id,))
            if in_db:
                # 若该电影已经在数据库中，则跳过一条
                print('Movie in db, skip one.')
                continue
        
        # 该条电影记录不在数据库中
        try:
            # 开始尝试爬取数据
            movie_data=get_movie_info(movie_id, sess, IMDB=True)
            # 爬取成功，未出异常，则将数据暂存与数据队列中
            data_queue.put(movie_data)
            print('Got movie:',movie_data['title'])
            i+=1
            # 第三阶段，判断队列中元素数量，如果到达某个数的整数倍便执行第三段代码，一次性插入
            if i%3==0:
                insertData(data_queue, conn, mode='m')

        except TimeoutError:
            print('Time out, skip one.')
            continue
        except TypeError as e:
            print(e) 
            if (e.__str__().startswith("'NoneType'")):
                print('403 detected. Process aborted.')
                break
            # 自动处理验证码
            if mplock:
                mplock.acquire()
                ok=verify(sess)
                mplock.release()
            else:
                ok=verify(sess)
            if not ok:
                break



def getMovieQueue(conn, maxNum=999999):
    sql="select watchedMovies from user where user_id<=%s"
    with conn.cursor() as cur:
        cur.execute(sql, (maxNum,))
        rawData=cur.fetchall()
    
    pat=re.compile(r"\'(\d+)\'")
    data=pat.findall(str(rawData))
    data=list(set(data))

    movie_queue=Queue()
    for i in data:
        movie_queue.put(i)
    print(len(data), 'items inserted to movie queue.')

    return (movie_queue, len(data))

    

def insertData(data_queue, conn, mode='u'):
    i=0

    if mode=='u':
        insert_sql="""insert into user
        (user_number, name, following_id, comments, rates, watchedMovies, wishMovies)
        values (%s, %s, %s, %s,%s, %s, %s)
        """
    elif mode=='m':
        insert_sql="""insert into movie
        (rate, title, id, directors, year, actors, type, countries, summary, runtime, imdb_rate)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    # 持续插入数据直到数据队列为空为止
    while not data_queue.empty():
        try:
            data=data_queue.get()
            if data is None:
                raise TypeError
            with conn.cursor() as cur:
                cur.execute(insert_sql, [str(i) for i in data.values()])
            
            if mode=='u':
                print('Inserted:', data['name'])
            elif mode=='m':
                print('Inserted:', data['title'])
            i+=1

        except TypeError as e:
            print(e)
            continue
        except Exception as e:
            if mode=='u':
                print('Fail to insert:', data['name'], e)
            elif mode=='m':
                print('Fail to insert:', data['title'], e)
            
    print(i, mode, 'inserted.')
        
        

