# coding: utf-8

import requests
import json
import re
import pickle
from PIL import Image
from bs4 import BeautifulSoup
from douban_spider.verti_code_preprocessing import transfertoCode

headers={
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate",
    'Content-Length': '0',
    "Connection": "keep-alive" 
}



def get_movie_info(movie_number, sess, IMDB=True):
    '''
    Input the number of the movie(the one used in Douban), 
    return the relating data of the movie in terms of a dictionary
    增加爬取电影时间runtime(分钟)
    增加爬取相关的imdb评分
    '''
    url='https://movie.douban.com/subject/{}/'.format(movie_number)
    if sess==None:
        sess=requests

    try:
        res=sess.get(url, headers=headers, timeout=3)

        print(res.status_code)
        if res.status_code!=200:
            if res.status_code==404:
                raise TimeoutError
            else:
                raise TypeError
        soup=BeautifulSoup(res.text, 'lxml')

        keyinfo=soup.find_all(type="application/ld+json")
        temp=json.loads(keyinfo[0].text, strict=False) if keyinfo else None
        # get title, rate, url, id, year through parsing html
        if temp:
            title=temp['name']
            rate=temp['aggregateRating']['ratingValue']
            rate=float(rate) if rate else None
        else:
            title=None
            rate=None

        id=temp['url'].split('/')[2]
        year=temp['datePublished'].split('-')[0]

        # get a piece of summary for the movie
        temp=soup.find_all(property='v:summary')
        summary=temp[0].text.strip() if temp else None

        # get the info about the movie's country
        temp=re.findall(r"制片国家/地区:</span>\s*([^\s<]*)", res.text)
        countries=temp[0].strip() if temp else None
        # temp=temp[4].nextSibling.split('/') if temp else None
        # for i in temp:
        #     if i.text=='制片国家/地区:'
        # countries=temp[0].strip() if temp else None


        # get the categories the movie is in
        temp=soup.find_all(property='v:genre')
        types=[i.text for i in temp]

        # get directors' namelist
        director=soup.find_all(rel='v:directedBy')
        directors=[i.text for i in director]

        # get actors' namelist
        actor=soup.find_all(rel='v:starring')
        actors=[i.text for i in actor]

        # get the runtime of the movie
        temp=soup.find_all(property='v:runtime')
        runtime=int(re.findall(r'\d+', temp[0].text)[0]) if temp else None

        # get IMDB ratings
        temp=soup.find(id='info').find(target="_blank",rel="nofollow")
        if temp and IMDB:
            temp=temp.text
            res_imdb=sess.get('https://www.imdb.com/title/'+temp, headers=headers)
            soup_imdb=BeautifulSoup(res_imdb.text, 'lxml')
            temp=soup_imdb.find(itemprop='ratingValue')
            imdb_rate=float(temp.text) if temp else None
        else:
            imdb_rate=None

        data={
            'rate':rate,
            'title':title,
            'id':id,
            'directors':directors,
            'year':year,
            'actors':actors,
            'type':types,
            'countries':countries,
            'summary':summary,
            'runtime':runtime,
            'imdb_rate':imdb_rate
        }
        return data

    except(AttributeError, TypeError) as e:
        raise TypeError(e)
    except:
        raise TimeoutError



def get_user_info(user_number, sess):
    '''
    Input the user number on Douban, 
    return the relating data in terms of a dictionary
    增加爬取“想看的电影”项
    增加信息：movieSeq表示观看该电影顺序，按时间降序排列
    不需要用户名字，这里抓取的仅仅为用户的user_number
    '''
    try:
        url='https://www.douban.com/people/{}/'.format(user_number)
        res=sess.get(url=url, headers=headers, timeout=3)
        if res.status_code!=200:
            raise TypeError
        soup=BeautifulSoup(res.text, 'lxml')

        name=soup.find('title').text.strip()

        # get following id
        follUrl='https://www.douban.com/people/{}/contacts'.format(user_number)
        follRes=sess.get(url=follUrl, headers=headers, timeout=3)
        follSoup=BeautifulSoup(follRes.text, 'lxml')
        temp=follSoup.find_all(class_='nbg')
        idPat=re.compile(r'e/(.*)/')
        following_id=[idPat.findall(i['href'])[0] for i in temp]

        # get the info about what movies he watched, what ratings he gave, what comments he made. 
        # For efficiency reasons, get at most 10 pages
        watUrl='https://movie.douban.com/people/{}/collect'.format(user_number)
        movieRate=dict()
        movieComment=dict()
        movieSeq=[]
        while True:
            count=0
            watRes=sess.get(url=watUrl, headers=headers, timeout=3)
            watSoup=BeautifulSoup(watRes.text, 'lxml')

            items=watSoup.find_all(class_='item')
            for i in items:
                title=re.findall(r'\d+', i.find(class_="title").find('a')['href'])[0]
                # ratings
                rateStr=i.find(class_='intro').nextSibling.nextSibling.find('span')['class'][0]
                # Some movies are not rated by the user
                movieRate[title]=None if len(rateStr)<=4 else int(re.findall(r'\d', rateStr)[0])

                # comments
                comment=i.find(class_='comment')
                if comment!=None:
                    movieComment[title]=comment.text
                
                # movie_id ordered by time (Descending)
                movieSeq.append(title)
            
            # if next page is available, move to the next page(6 pages at most, 15 items per page)
            nextPageAvailable=watSoup.find(rel='next')
            if count<6 and nextPageAvailable:
                watUrl='https://movie.douban.com'+nextPageAvailable['href']
                count+=1
            else:
                break
        
        # get information about the movies it wishes to watch(7 pages at most, 15 items per page)
        wishUrl='https://movie.douban.com/people/{}/wish'.format(user_number)
        movieSeq_wish=[]
        while True:
            count=0
            wishRes=sess.get(url=wishUrl, headers=headers, timeout=3)
            wishSoup=BeautifulSoup(wishRes.text, 'lxml')

            items=wishSoup.find_all(class_='item')
            for i in items:
                title=re.findall(r'\d+', i.find(class_="title").find('a')['href'])[0]  
                # movie_id ordered by time (Descending)
                movieSeq_wish.append(title)
            
            # if next page is available, move to the next page
            nextPageAvailable=wishSoup.find(rel='next')
            if count<6 and nextPageAvailable:
                wishUrl='https://movie.douban.com'+nextPageAvailable['href']
                count+=1
            else:
                break
    except(AttributeError, TypeError) as e:
        raise TypeError(e)
    except:
        raise TimeoutError
    
    data={ 
        'user_number':user_number,
        'name':name,
        'following_id':following_id,
        'comments':movieComment,
        'rates':movieRate,
        'watchedMovies':movieSeq,
        'wishMovies':movieSeq_wish
    }
    return data



def login(name, password):
    '''
    Fake a login session to circumvent the restriction, return the session object
    '''
    data={
        'name':name,
        'password':password,
        'remember':'false'
    }
    sess=requests.session()
    get=sess.post('https://accounts.douban.com/j/mobile/login/basic', data=data, headers=headers, timeout=3)

    # Print login state
    message=json.loads(get.text)
    print(message['description'])

    return sess



def verify(sess):
    '''
    An method to deal with the verification-needed case AUTOMATICALLY

    '''

    try:
        res=sess.get('https://www.douban.com/', headers=headers, timeout=3)
        soup=BeautifulSoup(res.text, 'lxml')
        while True:
            if (soup.find('h1')==None):
                print('Verified.')
                return True

            imgUrl=soup.find('img')['src']
            get_captcha=soup.find_all('input')
            captcha_id=get_captcha[2]['value']
            ck=get_captcha[0]['value']

            img=sess.get(imgUrl, headers=headers, timeout=3)
            with open('verif_code.png', 'wb') as f:
                f.write(img.content)

            # 由图片自动识别出验证码
            print('Analyzing verification code...')
            solution=transfertoCode('verif_code.png')

            data={
                'ck':ck,
                'captcha-solution':solution,
                'captcha-id':captcha_id,
                'original-url':'https://www.douban.com/'
            }

            res=sess.post('https://www.douban.com/misc/sorry', data=data)
            soup=BeautifulSoup(res.text, 'lxml')

    except Exception as e:
        print('Fail to verify!')
        print(e)
        if (e.__str__().startswith("'NoneType'")):
            print('403 detected. Process aborted.')
            return None

    return True

