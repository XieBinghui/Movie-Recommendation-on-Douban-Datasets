# coding: utf-8
import pymysql

class data2graph(object):
    def __init__(self):
        self.G=None
        pass

    def user_fromDB(self, user='root', passwd='123456', db='test', user_table_name='user',\
            autocommit=True):
        conn=pymysql.connect(user=user, password=passwd, db=db, autocommit=autocommit)
        with conn.cursor() as cur:
            # 获取电影在数据库中id和豆瓣id的对应信息
            num=cur.execute(r'''
                select name, following_id, user_id from {}
                '''.format(user_table_name))
            print(num, 'pairs got.')
            self.rawData=cur.fetchall()
            self.name2user_id=dict()
            for i in self.rawData:
                self.name2user_id[i[0]]=i[2]

    def edges2Text(self, filename):
        '''
        Write edges to txt for use of Graphtools.fromText
        Note that each id starts from 0.
        '''
        writelist=['N{}\tE{}\n']
        for item in self.rawData:
            for outedge in eval(item[1]):
                outid=self.name2user_id.get(outedge, -1)
                if outid==-1:
                    continue
                writelist.append(\
                    '{}\t{}\n'.format(item[2]-1, outid-1))
        with open(filename, 'w') as f:
            writelist[0]=writelist[0].format(len(self.rawData), len(writelist)-1)
            f.writelines(writelist)
            print('Data saved to file.')
        
            

            


    
        

if __name__=='__main__':
    data2graph=data2graph()
    data2graph.user_fromDB('root', '140014.Cjy', db='douban', user_table_name='user_original')
    data2graph.edges2Text('doubanUser_original.txt')

    