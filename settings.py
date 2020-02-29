# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/21 21:41
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESOURCE_DIR = os.path.join(PROJECT_DIR, "src")
SAMPLE_DATA_DIR = os.path.join(RESOURCE_DIR, "sample_data")
STANDARD_DATA_DIR = os.path.join(RESOURCE_DIR, "standard_data")


import logging
logging.basicConfig(level=logging.DEBUG)


import time
def calc_time(func):
	def wrapper(*args, **kwargs):
		st = time.time()
		f = func(*args, **kwargs)
		et = time.time()
		logging.debug("FUNC: {:15s}, TIME SPENT: {:.4f}".format(func.__name__, et-st))
		return f
	return wrapper