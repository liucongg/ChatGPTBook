# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_helper
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/2/26 15:40
"""
    文件说明：
            
"""

import re
from dirty_recognize import dirty_reg
import json


def remove_multi_symbol(text):
    """
    去除连续标点符号，并只保留第一个
    Args:
        text:

    Returns:

    """
    r = re.compile(r'([.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-])[.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-]+')
    text = r.sub(r'\1', text)
    return text


def remove_emojis(text):
    """
    去除表情符号
    Args:
        text:

    Returns:

    """
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def remove_html(text):
    """
    去除html
    Args:
        text:

    Returns:

    """
    reg = re.compile('<[^>]*>')
    text = reg.sub('', text).replace('\n', '').replace(' ', '')
    return text


def remove_dirty_sentence(dirty_obj, sentence):
    """
    判断句子中是否包含敏感词
    Args:
        dirty_obj:
        sentence:

    Returns:

    """
    if len(dirty_obj.match(sentence)) == 0:
        return False
    else:
        return True


def data_processing(path, dirty_obj, save_path):
    """
    清洗豆瓣夸夸群数据
    Args:
        path:
        dirty_obj:
        save_path:

    Returns:

    """
    s = ["大家来留言吧！我来夸你们", "求表扬", "有人夸我吗", "求安慰", "求祝福", "能被表扬吗", "求夸奖", "求鼓励",
         "来表扬我一下好吗", "求夸", "我好棒啊", "球表演", "求彩虹屁", "快来夸我嘛", "快来夸夸我", "再来夸一次哈哈"]

    n = 0
    data_dict = {}
    # 遍历文件中原始内容
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip().split("\t")
            # 判断数据为问句还是回答
            if "Q" in line[0]:
                q = "".join(line[1:])
                continue
            elif "A" in line[0]:
                # 当为回答时，去除不完整的问题和简单的回答
                a = "".join(line[1:])
                if "..." in q or "谢谢" in a:
                    continue
                # 判断是否包含敏感词，如果包含，则舍弃该数据
                if remove_dirty_sentence(dirty_obj, q):
                    continue
                if remove_dirty_sentence(dirty_obj, a):
                    continue
                # 对问题和回答去除html标签
                q = remove_html(q)
                a = remove_html(a)
                # 对问题和回答去除重复标点
                q = remove_multi_symbol(q)
                a = remove_multi_symbol(a)
                # 去除问题中，引用夸夸的部分
                for s_ in s:
                    q = q.replace(s_, "")
                # 对问题和回答去除标签符号标点
                q = remove_emojis(q)
                a = remove_emojis(a)
                # 清洗后如果问题和回答小于一定长度，舍弃该数据
                if len(q) <= 4 or len(a) <= 4:
                    continue
                else:
                    n += 1
                    if q not in data_dict:
                        data_dict[str(q)] = set()
                        data_dict[str(q)].add(a)
                    else:
                        data_dict[str(q)].add(a)
    # 将数据保存成模型训练所需要的数据格式
    fin = open(save_path, "w", encoding="utf-8")
    for key in data_dict.keys():
        for value in data_dict[key]:
            fin.write(json.dumps({"src_text": key, "tgt_text": value}, ensure_ascii=False) + "\n")
    print("total number of data:", n)
    # 121687
    fin.close()


if __name__ == "__main__":
    dirty_path = "data/dirty_words.txt"
    dirty_obj = dirty_reg(dirty_path)
    ori_path = "data/douban_kuakua_qa.txt"
    save_path = "data/train.json"
    data_processing(ori_path, dirty_obj, save_path)
