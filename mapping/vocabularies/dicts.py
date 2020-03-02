# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 1:07
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------
from general import *

import difflib


def __read_dict_file(file_path, multi_row=True):
	s = open(file_path, "r", encoding="utf-8")
	data = []
	if multi_row:
		for row in s.readlines():
			if row:
				data.append(row.strip())
	else:
		for alpha in s:
			if alpha:
				data.append(alpha)
	return data

@calc_time
def __get_the_match(input_str, words, match_type):
	best_tuple = (None, 0)
	for word in words:
		# quick_ratio()是通过集合方式去查找的，会乱序
		# 因此要使用ratio()方法稳健地比较，但会很慢，除非找到，最长可达二十秒
		sm = difflib.SequenceMatcher(a=input_str, b=word)
		score = sm.quick_ratio() if match_type == "quick" else sm.ratio()
		if score == 1:
			return (word, score)
		elif score > best_tuple[1]:
			best_tuple = (word, score)
	return best_tuple

def get_quick_match(input_str, words):
	logging.debug("Quick match is out of order! "
	              "If the result is unsatisfing, "
	              "please condiser using the func `get_best_match` "
	              "although it would be centainly slower...")
	return __get_the_match(input_str, words, match_type="quick")

def get_best_match(input_str, words):
	if len(words) > 100000:
		logging.debug("Please wait in patience since the words length achieves "
		              "{}...".format(len(words)))
	return __get_the_match(input_str, words, match_type="best")

cn_idioms = __read_dict_file("cn_idioms.txt", multi_row=True)
cn_chars  = __read_dict_file("cn_3500.txt", multi_row=False)
en_words  = __read_dict_file("en_words.txt", multi_row=True)

if __name__ == '__main__':
	matched_word, matched_score = get_best_match("望了成龙", cn_idioms)