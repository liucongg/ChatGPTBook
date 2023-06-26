# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: dirty_recognize
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:45
"""
    文件说明：
            
"""
from trie import Trie


class dirty_reg(object):
    def __init__(self, path):
        self.obj = Trie()
        self.build(path)

    def insert_new(self, word_list):
        word_list = [word.lower() for word in word_list]
        self.obj.insert(word_list)

    def build(self, path):
        f = open(path, "r", encoding="utf-8")
        for line in f:
            line = line.strip()
            if line:
                self.insert_new(line)

    def enumerateMatchList(self, word_list):
        word_list = [word.lower() for word in word_list]
        match_list = self.obj.enumerateMatch(word_list)
        return match_list

    def match(self, query):
        al = set()
        length = 0
        for indx in range(len(query)):
            index = indx + length
            match_list = self.enumerateMatchList(query[index:])
            if match_list == []:
                continue
            else:
                match_list = max(match_list)
                length = len("".join(match_list))
                al.add(match_list)
        return al
