# -*- coding: utf-8 -*- 
# @Time : 2023/4/12 23:47 
# @Author : JunkRoy 
# @E-mail: shenroy92@gmail.com
# @File : configs.py

import json
import argparse


def get_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        configs = json.load(f)
    args = argparse.ArgumentParser(description='glm pretrain')
    args_dict = vars(args)
    args_dict.update(configs)
    return args


if __name__ == '__main__':
    path = "./configs.json"
    args = get_config(path)
    print(args)
