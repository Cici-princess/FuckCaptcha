# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/26 15:57
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
import os
import time
import random
# random.seed(77)
import string
from enum import Enum
from pprint import pprint

import logging
logging.basicConfig(level=logging.DEBUG)

from PIL import Image, ImageDraw, ImageFont, ImageFilter


"""
验证码类型
"""
class CaptchaTypes(Enum):
    NUM             = "纯数字"
    ALPHA           = "纯英文"
    WORD            = "英文单词"
    NUM_ALPHA       = "数字+英文单词"
    CN_ALPHA        = "纯中文"
    CN_IDIOM        = "中文成语"
    CN_INVERSION    = "中文含倒置"
    CALCULATION     = "四则运算"


"""
颜色
"""
def _random_color(R_range, G_range, B_range):
    random_color = lambda x: random.randint(*x)
    return random_color(R_range), random_color(G_range), random_color(B_range)

def gen_random_bg_color():
    return _random_color((0, 135), (0, 135), (0, 135))

def gen_random_fg_color():
    return _random_color((100, 200), (100, 200), (100, 200))


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

"""
验证码
"""
class OurCaptcha:

    def __init__(self, char_type, char_text, IMG_WIDTH=320, IMG_HEIGHT=160, IMG_MODE="RGB"):
        self.char_type  = char_type
        self.char_text  = char_text
        self.char_cnt   = len(self.char_text)
        self.img_width  = IMG_WIDTH
        self.img_height = IMG_HEIGHT
        self.img_mode   = IMG_MODE
        self.img        = Image.new(self.img_mode, (self.img_width, self.img_height))  # 创建一张新图片
        self.img_draw   = ImageDraw.Draw(self.img)  # 创建绘图工具
        self.img_path   = None

    def _draw_bg(self):
        logging.debug("Initializing the background color...")
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                self.img_draw.point((i, j), gen_random_bg_color())

    def _draw_fg(self):
        logging.debug("Initializing the foreground text...")
        for char_seq in range(self.char_cnt):
            char_font    = use_random_font()
            char_color   = gen_random_fg_color()
            char_x       = int(self.img_width / (self.char_cnt + 1) * (char_seq + 0.5))
            char_y       = random.randint(int(self.img_height/5), int(self.img_height/3))
            self.img_draw.text((char_x, char_y), self.char_text[char_seq], char_color, char_font)
        
    def _add_noise(self):
        logging.debug("Adding some preset noises...")
        self._add_line(line_cnt=3)
        self.img.filter(ImageFilter.BLUR)   # 模糊处理


    def _add_line(self, line_cnt=3, joint="curve"):
        """
        (x_0, y_0): 线条的起始点坐标
        (x_1, y_1): 线条的终止点坐标
        l_l：线条的长度
        l_w: 线条的宽度
        joint: 线条是直线还是曲线
        :param line_cnt:
        :return:
        """
        for i in range(line_cnt):
            x_0 = random.randint(0, int(self.img_width/3))
            y_0 = random.randint(int(self.img_height/3), int(self.img_height/3*2))
            l_l = random.randint(int(self.img_width/2), int(self.img_width))
            l_w = random.randint(1, 3)
            x_1 = x_0 + l_l
            y_1 = y_0 + random.randint(-int(self.img_height/3), int(self.img_height/3))
            self.img_draw.line(((x_0, y_0), (x_1, y_1)), fill=gen_random_bg_color(), width=l_w, joint=joint)

    def create_image(self):
            logging.debug("Creating one image...")
            self._draw_bg()
            self._draw_fg()
            self._add_noise()
        
    def show_image(self):
        logging.debug("Showing the image, "
                      "please close the image viewer "
                      "before continuing to the following procedures if necessary.")
        self.img.show()

    def save_image(self, file_name=None, file_dir_path=None):
        if not file_dir_path:
            file_dir_path = "SavedCaptcha"
            if not os.path.exists(file_dir_path):
                os.mkdir(file_dir_path)
        if not file_name:
            file_name = "{}.jpg".format(int(time.time()))
        self.img_path = os.path.abspath(os.path.join(file_dir_path, file_name))
        self.img.save(self.img_path)
        logging.debug("Successfully saved image to path {}".format(self.img_path))

    @property
    def data_info(self):
        return {"type": self.char_type, "text": self.char_text, "path": self.img_path}


if __name__ == '__main__':

    # 这里放一些你需要的字体格式名，windows路径为 C:\Windows\Fonts
    FONT_TYPES      = ['simhei.ttf', 'simkai.ttf']
    CAPTCHA_TYPE    = CaptchaTypes.CN_ALPHA
    CHAR_CNT        = 4
    captcha_text    = gen_captcha_text(char_type=CAPTCHA_TYPE, char_cnt=CHAR_CNT)
    logging.info("生成文字：{}".format(captcha_text))

    our_captcha = OurCaptcha(char_type=CAPTCHA_TYPE, char_text=captcha_text)
    our_captcha.create_image()
    our_captcha.show_image()
    our_captcha.save_image()
    pprint(our_captcha.data_info)
