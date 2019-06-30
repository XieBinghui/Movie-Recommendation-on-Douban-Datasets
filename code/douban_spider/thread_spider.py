import time

def userThread(user_queue, maxNum, session):
    '''
    Open a thread, and loop:
        get a userid from user_queue, get its info from internet
        put its following id into user_queue, and its data into data_queue
        until processed number reaches maxNum
    '''
    i=0
    total=0
    global data_queue

    while i<maxNum:
        user_id=user_queue.get()
        # print('getting data')
        try:
            user_data=get_user_info(user_id, session)
            data_queue.put(user_data)

        except TypeError:
            # 爬数据超过一定量会出现的验证码问题
            print('Need verification-user')
            time.sleep(4)
            # verify(sess)
        except:
            print('fail to get data', i)
            continue

        i+=1
        # print('data queue size', data_queue.qsize())
        print('user queue size', user_queue.qsize())
        
        if user_data==None:
            time.sleep(4)
            continue
        
        if total<maxNum:
            for item in user_data['following_id']:
                user_queue.put(item)
            total+=len(user_data['following_id'])
    
    print('userThread exit')


def ioThread(sql, maxNum, conn, Input=True):
    '''
    Open a thread, and loop:
        get data from data_queue if not empty, write(if Input=True) it to the database [else read one from database]
        if the queue is empty and processed number hasn't reached maxNum, keep waiting
        Exit when processed number reaches maxNum
    '''
    i=0
    global data_queue

    if Input:
        while i<maxNum:
            while data_queue.empty():
                time.sleep(0.1)
            data=data_queue.get()
            try:
                execSQL(sql, conn, data=data)
                print('Inserted:', data['name'])
                i+=1
            except TypeError:
                verify(sess)
            except Exception:
                print('Fail to insert:', data['name'])
    else:
        pass # need to implement Output part
    
    print('ioThread exit')