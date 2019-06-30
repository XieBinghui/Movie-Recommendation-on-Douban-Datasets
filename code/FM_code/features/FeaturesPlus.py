import pandas as pd
import numpy as np
import random

from sqlalchemy import create_engine


def connectSql():
    engine = create_engine('mysql://root:bigdata10@localhost:3306/total', convert_unicode=True, encoding='utf-8',
                           connect_args={"charset": "utf8"})
    df_movie = pd.read_sql('movie_', engine, index_col='movie_id')
    df_user = pd.read_sql('user_', engine, index_col='user_id')
    return df_movie, df_user


def preprocess(df_movie, df_user):
    print('Preprocessing...')

    # Idx = 1,2,3... Id = Movie Id
    # 记录Movie的Id到Idx 方便后面对接用户
    # 记录电影的所有Type
    MovieId2Idx = {}
    AllType = []
    m_idx = 0
    for idx, row in df_movie.iterrows():
        MovieId2Idx[row['id']] = m_idx
        m_idx += 1
        AllType.extend(row['type'].split(','))
    AllType = list(set(AllType))
    TypeDict = {AllType[i]: i for i in range(len(AllType))}

    # 记录用户名到ID
    UserIdx2following ={}
    UserId2Idx = {}
    # for idx, row in df_user.iterrows():
    #     UserId2Idx[row['name']] = idx
    idx = 0
    for i, row in df_user.iterrows():
        num = 0
        movie = eval(row['rates'])
        for id in movie.keys():
            if movie[id]:
                num = 1
        if num:
            if row['name'] not in UserId2Idx.keys():   # qu chong
                UserId2Idx[row['name']] = idx
                idx += 1

    for i, row in df_user.iterrows():
        if row['name'] in UserId2Idx.keys():
            idx = UserId2Idx[row['name']]
            usr_following = row['following_id']
            following_id = set(usr_following) & set(UserId2Idx.keys())
            UserIdx2following[idx] = [UserId2Idx[id] for id in following_id]

    following_rates = np.zeros(shape=(len(UserId2Idx), len(df_movie)))
    for i, row in df_user.iterrows():
        if row['name'] in UserId2Idx.keys():
            idx = UserId2Idx[row['name']]
            rates_id = eval(row['rates'])
            for id in rates_id.keys():
                if id in MovieId2Idx:
                    following_rates[idx][MovieId2Idx[id]] = rates_id[id] # following_rates is rates

    # print(UserId2Idx)
    # print(MovieId2Idx)
    # print(UserIdx2following)
    # print(following_rates)

    return UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, TypeDict


def pos_sample(df_movie, df_user, UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, TypeDict, bias, path):
    file = open(path + '.pos', 'w')

    for i, (_, row) in enumerate(df_user.iterrows()):
        if row['name'] in UserId2Idx.keys():
            ur_idx = UserId2Idx[row['name']]
            print('Pos Sampling Processing: ' + str(ur_idx))

            # 对每一个用户 将他评分过的电影的Id:rate的字典转成Idx:rate
            rates = eval(row['rates'])
            MovieIdx2rate = {}
            for id in rates.keys():
                if id in MovieId2Idx:
                    if rates[id]:
                        MovieIdx2rate[MovieId2Idx[id]] = rates[id]

            # 利用idx去使用dataframe
            # 取电影的year, average rate, type信息
            MovieIdx2yr = {}
            MovieIdx2avrate = {}
            MovieIdx2type = {}
            for idx in MovieIdx2rate.keys():
                MovieIdx2yr[idx] = df_movie.iloc[idx]['year']
                try:
                    if df_movie.iloc[idx]['rate'] == 'None':
                        print(df_movie.iloc[idx]['rate'])
                        df_movie.iloc[idx]['rate'] = 0
                except KeyError:
                    print("The problem is about idx or iloc")
                MovieIdx2avrate[idx] = float(df_movie.iloc[idx]['rate']) / 2  # /2是因为有10分和五星的区别
                tps = df_movie.iloc[idx]['type'].split(',')
                MovieIdx2type[idx] = [TypeDict[tp] for tp in tps]

            # 对该用户评价过的每一个电影 生成一个特征
            for idx in MovieIdx2rate.keys():
                # 将正在生成特征的这个电影的用户评分换成电影的平均分数

                rates = {idx: MovieIdx2rate[idx] for idx in MovieIdx2rate.keys()}
                # feature 2:
                rates[idx] = 0
                # feature 3:
                rates[idx] = MovieIdx2avrate[idx]

                features = []
                # add ground truth
                if MovieIdx2rate[idx] == None:
                    continue
                else:
                    features.append(MovieIdx2rate[idx])
                # add user
                features.append(str(ur_idx) + ':1')  # features.append(ur_idx)
                # add item
                features.append(str(bias[0] + int(idx)) + ':1')     # features.append(idx)
                # # add rates
                # for mv_idx in rates.keys():
                #     features.append(str(bias[1] + int(mv_idx)) + ':' + str(rates[mv_idx]))

                # add movie year
                if MovieIdx2yr[idx] != '':
                    features.append(str(bias[2]) + ':' + MovieIdx2yr[idx])
                else:
                    features.append(str(bias[2]) + ':0')

                # add movie type
                for i in MovieIdx2type[idx]:
                    features.append(str(bias[2] + i) + ':' + '1')
                #
                # feature 4:
                # add user following id
                # for usr in UserIdx2following[ur_idx]:
                #     features.append(str(bias[2] + int(usr)) + ':' + '1')

                # # feature 5:
                # # add user following rates
                # follw_usr_mv_2d = np.argwhere(following_rates[UserIdx2following[ur_idx]].mean(axis=0)>0)
                # follw_usr_mv = follw_usr_mv_2d.flatten()
                # follw_usr_mv_rates = following_rates[UserIdx2following[ur_idx]].mean(axis=0)[follw_usr_mv_2d.reshape(None,)].flatten()
                # for mv in range(len(follw_usr_mv)):
                #     features.append(str(bias[3] + int(follw_usr_mv[mv])) + ':' + str(follw_usr_mv_rates[mv]))

                to_sort = features[1:]
                to_sort.sort(key=lambda x: int(x.split(':')[0]))
                features[1:] = to_sort
                features[0] = str(features[0])
                file.write(' '.join(features) + '\n')

    file.close()


def neg_sample(df_movie, df_user, UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, TypeDict, bias, path):
    file = open(path + '.neg', 'w')

    for i, (_, row) in enumerate(df_user.iterrows()):
        if row['name'] in UserId2Idx.keys():
            ur_idx = UserId2Idx[row['name']]
            print('Neg Sampling Processing: ' + str(ur_idx))

            # 对每一个用户 将他评分过的电影的Id:rate的字典转成Idx:rate
            rates = eval(row['rates'])
            MovieIdx2rate = {}
            for id in rates.keys():
                if id in MovieId2Idx:
                    if rates[id]:
                        MovieIdx2rate[MovieId2Idx[id]] = rates[id]

            # 生成没有评价过的电影列表
            unrates = set(range(len(df_movie))) - set(MovieIdx2rate.keys())

            # 对该用户评价过的每一个电影
            # 利用idx去索引dataframe
            # 取电影的year, average rate, type信息
            MovieIdx2yr = {}
            MovieIdx2avrate = {}
            MovieIdx2type = {}
            for idx in unrates:
                MovieIdx2yr[idx] = df_movie.iloc[idx]['year']
                if df_movie.iloc[idx]['rate'] == 'None':
                    df_movie.iloc[idx]['rate'] = 0
                MovieIdx2avrate[idx] = float(df_movie.iloc[idx]['rate'])/2
                tps = df_movie.iloc[idx]['type'].split(',')
                MovieIdx2type[idx] = [TypeDict[tp] for tp in tps]

            # 对该用户没评价过的每一个电影 生成一个特征
            sample_count = 0
            for idx in unrates:
                sample_count += 1
                if sample_count % 400 == 0:
                    features = []
                    # add ground truth
                    features.append('0')
                    # add user
                    features.append(str(ur_idx) + ':1')  # features.append(ur_idx)
                    # add item
                    features.append(str(bias[0] + int(idx)) + ':1')     # features.append(idx)
                    # feature 2:
                    # add rates
                    # for mv_idx in MovieIdx2rate.keys():
                    #     features.append(str(bias[1] + int(mv_idx)) + ':' + str(MovieIdx2rate[mv_idx]))
                    # # feature 3:
                    # # 加入这个没评价过的电影的平均分数
                    # features.append(str(bias[1] + int(idx)) + ':' + str(float(df_movie.loc[idx]['rate'])/2))

                    # add movie year
                    if MovieIdx2yr[idx] != '':
                        features.append(str(bias[2]) + ':' + MovieIdx2yr[idx])
                    else:
                        features.append(str(bias[2]) + ':0')

                    # add movie type
                    for i in MovieIdx2type[idx]:
                        features.append(str(bias[2] +1 + i) + ':' + '1')
                    #
                    # feature 4:
                    # add user following id
                    # for usr in UserIdx2following[ur_idx]:
                    #     features.append(str(bias[2] + int(usr)) + ':' + '1')

                    # feature 5:
                    # add user following rates
                    # follw_usr_mv_2d = np.argwhere(following_rates[UserIdx2following[ur_idx]].mean(axis=0) > 0)
                    # follw_usr_mv = follw_usr_mv_2d.flatten()
                    # follw_usr_mv_rates = following_rates[UserIdx2following[ur_idx]].mean(axis=0)[
                    #     follw_usr_mv_2d.reshape(None, )].flatten()
                    # for mv in range(len(follw_usr_mv)):
                    #     features.append(str(bias[3] + int(follw_usr_mv[mv])) + ':' + str(follw_usr_mv_rates[mv]))

                    to_sort = features[1:]
                    to_sort.sort(key=lambda x: int(x.split(':')[0]))
                    features[1:] = to_sort
                    features[0] = str(features[0])
                    file.write(' '.join(features) + '\n')

    file.close()

def my_write(path, list):
    print("Hold on plz. I'm writing to " + path)
    file = open(path, 'w')
    for line in list:
        file.write(line)
    file.close()

def split_dataset(path):
    pos_file = open(path + '.pos', 'r')
    neg_file = open(path + '.neg', 'r')

    pos_line = []
    ur_range_pos = {}    # 记录每一个用户对应的正样本长度区间
    before_line, ur = 0, '0'
    for (num, line) in enumerate(pos_file.readlines()):
        pos_line.append(line)

        if ur != line.split()[1].split(':')[0] and (num-1) - before_line > 2:
            ur_range_pos[ur] = [before_line, num-1]
            before_line = num
            ur = line.split()[1].split(':')[0]
        elif num - before_line > 2 and ur == '3557':
            ur_range_pos[ur] = [before_line, num]      # 加入最后一个用户
    pos_file.close()

    neg_line = []
    ur_range_neg = {}  # 记录每一个用户对应的负样本长度区间
    before_line, ur = 0, '0'
    for (num, line) in enumerate(neg_file.readlines()):
        neg_line.append(line)

        if ur != line.split()[1].split(':')[0] and (num-1) - before_line > 2:
            ur_range_neg[ur] = [before_line, num - 1]
            before_line = num
            ur = line.split()[1].split(':')[0]
        elif num - before_line > 2 and ur == '3557':
            ur_range_neg[ur] = [before_line, num]      # 加入最后一个用户
    neg_file.close()

    # print(len(ur_range_pos))   # 989
    # print(len(ur_range_neg))   # 1000

    test = []
    valid = []
    train = []

    for ur in ur_range_pos.keys():
        print('Split Dataset Processing: ' + ur)
        posL, posR = ur_range_pos[ur]
        negL, negR = ur_range_neg[ur]

        posUr = [x for x in pos_line[posL:posR]]   # 生成新列表 防止后面的del操作改变pos_line
        negUr = [x for x in neg_line[negL:negR]]

        random.seed(55)
        tst, vld = random.sample(range(len(posUr)), 2)
        test.append(posUr[tst])
        valid.append(posUr[vld])
        del posUr[tst]
        del posUr[vld-1]         # 由于此时已经删除了一个元素 后面所有元素都往前移 所以角标-1
        train.extend(posUr)      # 加入除test和valid之外的所有正样本

        k, j = 99, 4
        # print(len(negUr))
        if len(negUr) > 99:
            tst = random.sample(range(len(negUr)), k)       # 每个用户取了k个负样本加入测试集和验证集
            vld = random.sample(range(len(negUr)), k)
            trn = random.sample(range(len(negUr)), j)       # 每个用户取了j个负样本加入训练集
        else:
            k, j = len(negUr), 4
            tst = random.sample(range(len(negUr)), k)  # 每个用户取了k个负样本加入测试集和验证集
            vld = random.sample(range(len(negUr)), k)
            trn = random.sample(range(len(negUr)), j)  # 每个用户取了j个负样本加入训练集

        for i in range(len(negUr)):
            if i in tst:
                test.append(negUr[i])
            if i in vld:
                valid.append(negUr[i])
            if i in trn:
                train.append(negUr[i])

    my_write(path + '.train', train)
    my_write(path + '.test', test)
    my_write(path + '.validation', valid)

if __name__ == '__main__':
    df_movie, df_user = connectSql()
    print(len(df_user))
    # embed = np.load('douban_deepwalk_us_64.npy')
    print(embed.shape)
    df_user = df_user.drop_duplicates(['name'], keep='first')
    df_movie = df_movie.drop_duplicates(['id'], keep='first')
    # df_user = df_user
    # print(len(df_user))
    df_movie = df_movie[:20000]

    # UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, typeDict = preprocess(df_movie, df_user)
    # usr_len = len(df_user)
    # movie_len = len(df_movie)
    # # print(usr_len)
    # # print(movie_len)
    # bias = [usr_len, usr_len+movie_len, usr_len+2*movie_len, usr_len*2+2*movie_len, usr_len*2+3*movie_len]
    # path = './plus/douban'    # 40381 douban.pos, 959619 douban.neg
    # pos_sample(df_movie, df_user, UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, typeDict, bias, path)
    # neg_sample(df_movie, df_user, UserId2Idx, MovieId2Idx, UserIdx2following, following_rates, typeDict, bias, path)
    # split_dataset(path)