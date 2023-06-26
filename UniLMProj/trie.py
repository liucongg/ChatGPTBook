# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: trie
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:46
"""
    文件说明：
            
"""


class Trie:
    def __init__(self):
        self.root = {}
        self.end = -1

    def insert(self, word):
        curNode = self.root

        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.end] = True

    def search(self, word):
        curNode = self.root

        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]

        if not self.end in curNode:
            return False
        return True

    def startsWith(self, pcurNodeix):
        curNode = self.root

        for c in pcurNodeix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        return True

    def get_start(self, prefix):
        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.extend(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list

        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)

    def enumerateMatch(self, word, space=""):
        matched = []
        while len(word) > 1:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched
