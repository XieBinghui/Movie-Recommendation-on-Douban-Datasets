# Movie-Recommendation-on-Douban-Datasets

#### 成员：

[谢炳辉](https://github.com/XieBinghui) [刘佰川](https://github.com/chuanchuanchuan) [程嘉扬](https://github.com/loginaway)

#### 报告

更多细节可以参考report

our report is [here](https://github.com/XieBinghui/Movie-Recommendation-on-Douban-Datasets/blob/master/Movie%20Recommendation%20on%20Douban%20Datasets.pdf)

#### 数据集格式：

豆瓣项目的数据集为sql格式，里面共有两个表，movie和user。请使用mysql进行读取和操作，以避免出现不必要的麻烦，推荐安装好mysql后使用SQLyog图形管理工具直接导入。

movie表：

​	number：该数据集中电影的编号，从1到1000

​	rate：电影在豆瓣的平均分

​	title：电影名字

​	url：电影链接

​	id：电影在豆瓣中的编号

​	directors：电影的导演

​	year：电影首播年份

​	actors：主演

​	type：电影类型

​	countries：制片国家、地区

​	summary：电影剧情简介

user表：

​	user_id：该数据集中用户的编号，从1到1000

​	name：用户在豆瓣中的编号

​	rates：用户看过的所有电影编号以及该用户给出的评分

​	following_id：该用户关注的所有用户的编号

​	comments：该用户对看过的一些电影的部分评论

#### 致谢：

感谢优秀的开源项目和老师以及助教的帮助

https://github.com/swjcpy/hybrid-neural-recommender-system

https://github.com/hexiangnan/neural_factorization_machine

https://github.com/AaronHeee/FDU-Recommender-Systems-for-Douban-Movie

https://github.com/hwwang55/RippleNet

https://github.com/kyrre/pmf

