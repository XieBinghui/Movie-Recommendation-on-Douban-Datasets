# coding: utf-8

from PIL import Image
import pytesseract
from spellchecker import SpellChecker
 
WHITE = (255,255,255)
BLACK = (0,0,0)

def pre_concert(img):
    #对图片做预处理，去除背景
    width,height = img.size
    threshold = 30
    for i in range(0,width):
        for j in range(0,height):
            p = img.getpixel((i,j))#抽取每个像素点的像素
            r,g,b = p
            if r > threshold or g > threshold or b > threshold:
                img.putpixel((i,j),WHITE)
            else:
                img.putpixel((i,j),BLACK)
    # img.show()
    img.save("pre_fig.jpg")
    return
 
 

def remove_noise(self, window=1):
    #对去除背景的图片做噪点处理
    if window == 1:
        window_x = [1,0,0,-1,0]
        window_y = [0,1,0,0,-1]
    elif window == 2:
        window_x = [-1,0,1,-1,0,1,1,-1,0]
        window_y = [-1,-1,-1,1,1,1,0,0,0]
 
    width,height = self.size
    for i in range(width):
        for j in range(height):
            box = []
 
            for k in range(len(window_x)):
                d_x = i + window_x[k]
                d_y = j + window_y[k]
                try:
                    d_point = self.getpixel((d_x,d_y))
                    if d_point == BLACK:
                        box.append(1)
                    else:
                        box.append(0)
                except IndexError:
                        self.putpixel((i,j),WHITE)
                        continue
 
            box.sort()
            if len(box) == len(window_x):
                mid = box[int(len(box)/2)]
                if mid == 1:
                    self.putpixel((i,j),BLACK)
                else:
                    self.putpixel((i,j),WHITE)
    # self.show()
    self.save("mov_noise_fig.jpg")
    return
 
 
 
def image_to_string(img,config='-psm 8'):
    try:
        result = pytesseract.image_to_string(img,lang='eng',config=config)
        result = result.strip()
        return result.lower()
    except:
        return None
 


def transfertoCode(imgAddr):
    # 根据验证码生成估计的验证字符串corr
    img=Image.open(imgAddr)
    pre_concert(img)
    remove_noise(img,2)
    img=Image.open('mov_noise_fig.jpg')
    #产生字符串
    s=pytesseract.image_to_string(img)
    #拼写纠错
    spell=SpellChecker()
    corr=spell.correction(s)
    return corr
