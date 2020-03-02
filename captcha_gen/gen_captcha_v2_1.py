# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/26 15:57
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

import time
import random
# random.seed(77)
import string
from enum import Enum
from pprint import pprint

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

from settings import *
IMG_SAVE_DIR = os.path.join(DATA_DIR, "captcha_gen_v2")
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# 这里放一些你需要的字体格式名，windows路径为 C:\Windows\Fonts
FONT_TYPES = ['simhei.ttf', 'simkai.ttf']

IMG_SIZE = (320, 160)


"""
验证码类型
"""
class CaptchaTypes(Enum):
    NUM             = 1
    ALPHA           = 2
    WORD            = 3
    NUM_ALPHA       = 4
    CN_ALPHA        = 5
    CN_IDIOM        = 6
    CN_INVERSION    = 7
    CALCULATION     = 8


"""
颜色
"""
def _random_color(R_range, G_range, B_range):
    random_color = lambda x: random.randint(*x)
    return random_color(R_range), random_color(G_range), random_color(B_range)

def gen_random_bg_color():
    return _random_color((50, 150), (50, 150), (50, 150))

def gen_random_fg_color():
    return _random_color((0, 100), (0, 100), (0, 100))


"""
字体
"""    
def use_random_font(font_size=70):
    return ImageFont.truetype(random.choice(FONT_TYPES), font_size) # 选择字体

def _gen_random_char(char_type):
    ALLOWED_CHAR_TYPES = [
        CaptchaTypes.NUM,
        CaptchaTypes.ALPHA,
        CaptchaTypes.NUM_ALPHA,
        CaptchaTypes.CN_ALPHA,
        CaptchaTypes.CALCULATION
    ]
    assert char_type in ALLOWED_CHAR_TYPES, \
        "暂不支持此类字符 {} 的生成，请确认：{}".format(char_type, ALLOWED_CHAR_TYPES)

    if char_type == CaptchaTypes.NUM:
        return random.choice(string.digits)

    if char_type == CaptchaTypes.ALPHA:
        return random.choice(string.ascii_letters)

    if char_type == CaptchaTypes.NUM_ALPHA:
        return random.choice(string.digits + string.ascii_letters)

    if char_type == CaptchaTypes.CN_ALPHA:
        return chr(random.randint(0x4e00, 0x9fbf))

    # TODO: 运算规则的开发
    if char_type == CaptchaTypes.CALCULATION:
        raise Exception("还在开发中，敬请等待...")

def gen_captcha_text(char_type, char_cnt):
    logging.debug("Generating {} chars of type {}".format(char_cnt, char_type))
    return "".join(_gen_random_char(char_type) for i in range(char_cnt))


class ImageHandle:

    def __init__(self, img_width, img_height, img_mode="RGBA"):
        self.img_width = img_width
        self.img_height = img_height
        self.img_mode   = img_mode
        self.img = Image.new(self.img_mode, (self.img_width, self.img_height))

    def add_text(self, char_text, font_size=50):
        char_cnt = len(char_text)
        logging.debug("Initializing the foreground text...")
        for char_seq in range(char_cnt):
            char_font    = use_random_font(font_size)
            char_color   = gen_random_fg_color()
            char_x       = int(self.img_width / (char_cnt + 1) * (char_seq + 0.5))
            char_y       = random.randint(int(self.img_height/5), int(self.img_height/3))
            ImageDraw.Draw(self.img).text((char_x, char_y), char_text[char_seq], char_color, char_font)
        return self

    def add_blur(self):
        self.img = self.img.filter(ImageFilter.BLUR)  # 模糊处理
        return self

    def add_bg_with_noise(self):
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                ImageDraw.Draw(self.img).point((i, j), gen_random_bg_color())
        return self

    def add_line(self, line_cnt=3, joint="curve"):
        """
        (x_0, y_0): 线条的起始点坐标
        (x_1, y_1): 线条的终止点坐标
        l_l：线条的长度
        l_w: 线条的宽度
        joint: 线条是直线还是曲线
        :param line_cnt:
        :return:
        """
        # TODO: 替换成rot_img函数

        for i in range(line_cnt):
            x_0 = random.randint(0, int(self.img_width/3))
            y_0 = random.randint(int(self.img_height/3), int(self.img_height/3*2))
            l_l = random.randint(int(self.img_width/2), int(self.img_width))
            l_w = random.randint(1, 3)
            x_1 = x_0 + l_l
            y_1 = y_0 + random.randint(-int(self.img_height/3), int(self.img_height/3))
            ImageDraw.Draw(self.img).line(((x_0, y_0), (x_1, y_1)), fill=gen_random_bg_color(), width=l_w, joint=joint)
        return self

    def add_warp(self, max_angle=15):
        random_angle = random.randint(-max_angle, max_angle) # 随机旋转-10-10角度
        self.img = self.img.rotate(random_angle)

        # 图形扭曲参数
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500]

        # 创建扭曲
        self.img = self.img.transform(self.img.size, Image.PERSPECTIVE, params)



        return self

"""
验证码
"""
class OurCaptcha:

    def __init__(self, img_width=320, img_height=160, img_mode="RGBA"):
        self.img_width = img_width
        self.img_height = img_height
        self.img_mode   = img_mode

    def create_image(self, char_text, line_cnt=3, font_size=50, max_rotate_angle=15):
        self.bg = ImageHandle(self.img_width, self.img_height, self.img_mode)
        self.fg = ImageHandle(self.img_width, self.img_height, self.img_mode)
        logging.debug("Creating one image...")
        self.bg.add_bg_with_noise()
        self.fg\
            .add_text(char_text, font_size)\
            .add_warp(max_rotate_angle)\
            .add_blur()\
            .add_line(line_cnt)

        self.bg.img.alpha_composite(self.fg.img)
        self.img = self.bg.img
        return self.img
        
    def show_image(self, img_title, use_pillow=False, use_cv=False):
        if use_pillow or use_cv:
            logging.debug("Showing the image, "
                      "please close the image viewer "
                      "before continuing to the following procedures if necessary.")
            if use_pillow:
                import matplotlib.pyplot as plt
                plt.title(img_title)
                plt.imshow(self.img)
                plt.show()
            if use_cv:
                import cv2
                cv2.imshow(img_title, np.array(self.img))
                cv2.waitKey()
        else:
            logging.warning("Since you do not enable open method neither USE_PILLOW nor USE_CV,"
                            "the image can't be shown!")

    def save_image(self, img, file_path):
        img.save(file_path)
        logging.debug("Successfully saved image to path {}".format(file_path))

    def __gen_one_captcha(self):
        captcha_text    = gen_captcha_text(char_type=CAPTCHA_TYPE, char_cnt=CHAR_CNT)
        img_name        = "{}_{}.png".format(captcha_text, time.time())
        img_path    = os.path.join(IMG_SAVE_DIR_BY_TYPE, img_name)
        img = self.create_image(char_text=captcha_text)
        return img, captcha_text, img_path

    def multi_save_image(self):
        while True:
            if not self.q.empty():
                img, img_text, img_path = self.q.get()
                if img is not None:
                    self.save_image(img, file_path=img_path)
                    logging.info({
                        "type": CAPTCHA_TYPE,
                        "text": img_text,
                        "path": img_path,
                    })
                else:
                    break
            else:
                time.sleep(1)

    def multi_gen_captchas(self, gen_cnt, ):
        from threading import Thread
        from queue import Queue
        self.q = Queue()
        self.p_save_img = Thread(target=self.multi_save_image)
        self.p_save_img.start()

        for i in range(gen_cnt):
            img_info = self.__gen_one_captcha()
            self.q.put(img_info)
        else:
            self.q.put((None, None, None))

        self.p_save_img.join()
        logging.info("Finished!")


if __name__ == '__main__':
    CAPTCHA_TYPE = CaptchaTypes.NUM
    IMG_SAVE_DIR_BY_TYPE = os.path.join(IMG_SAVE_DIR, "Type_{}".format(CAPTCHA_TYPE.value))
    os.makedirs(IMG_SAVE_DIR_BY_TYPE, exist_ok=True)
    CHAR_CNT = 1
    GEN_CNT  = 100

    our_captcha = OurCaptcha(*IMG_SIZE)
    our_captcha.multi_gen_captchas(gen_cnt=100)
