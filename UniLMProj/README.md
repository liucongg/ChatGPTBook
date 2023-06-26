本项目为书籍《ChatGPT原理与实战：大型语言模型的算法、技术和私有化》中第3章《预训练语言模型》实战部分代码-基于夸夸闲聊数据的UniLM模型实战。

## 项目简介

针对夸夸闲聊数据集，利用UniLM模型进行模型训练及测试，更深入地了解预训练语言模型的使用方法，完成一个生成式闲聊机器人任务。

项目主要结构如下：

- data 存放数据的文件夹
    - dirty_word.txt 敏感词数据
    - douban_kuakua_qa.txt 原始语料
    - sample.json 处理后的语料样例
- kuakua_robot_model 已训练好的模型路径
    - config.json
    - pytorch_model.bin
    - vocab.txt
- pretrain_model UniLm预训练文件路径
    - config.json
    - pytorch_model.bin
    - vocab.txt
- chatbot.py 模型推理文件
- configuration_unilm.py UniLm配置文件
- data_helper.py 数据预处理文件
- data_set.py 数据类文件
- modeling_unilm.py UniLm模型文件
- train.py 模型训练文件
- dirty_recognize.py 敏感词检测文件

注意：由于GitHub不方便放模型文件，因此data文件中douban_kuakua_qa.txt文件、kuakua_robot_model文件夹和pretrain_model文件夹中的模型bin文件，请从百度云盘中下载。

| 文件名称 | 下载地址 | 提取码 |
| --- |--- |---|
| pretrain_model | [百度云](https://pan.baidu.com/s/1sA9xf5ikjwDqXj9b7o4G8g) | 7h4a |
| kuakua_robot_model |[百度云](https://pan.baidu.com/s/1DFw4P8o1iVBeUgKKVufrFA) |j954|
| data |[百度云](https://pan.baidu.com/s/1PQKmcqLmN-x959E9LnSCQw) |dd0r|

## 环境配置

模型训练或推理所需环境，请参考requirements.txt文件。

## 数据处理

数据预处理需要运行data_helper.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
python3 data_helper.py
```

注意：如果需要修改数据生成路径或名称，请修改data_helper.py文件147-150行，自行定义。

## 模型训练

模型训练需要运行train.py文件，会自动生成output_dir文件夹，存放每个epoch保存的模型文件。

命令如下：

```shell
python3 train.py --device 0 \
                 --data_dir "data/" \
                 --src_file "train.json" \
                 --model_name_or_path "pretrain_model/" \
                 --max_seq_length 256 \
                 --train_batch_size 16 \
                 --num_train_epochs 10  
```

注意：当服务器资源不同或读者更换数据等时，可以在模型训练时修改响应参数，详细参数说明见代码或阅读书3.5.4小节。

模型训练示例如下：

![img.png](images/3_1.png)

模型训练阶段损失值变化如下：
![img.png](images/3_2.png)

## 模型推理

模型训练需要运行chatbot.py文件，可以采用项目中以提供的模型，也可以采用自己训练后的模型。

命令如下：

```shell
python3 chatbot.py --device 0 --topk 3 --topp 0.95 --max_len 32
```

注意：如果修改模型路径，请修改--model_path参数。

模型推理示例如下：
![img.png](images/3_3.png)

```text
USER:工作不开心
CHAT-BOT:我也是，不过现在开始努力找工作啦！一起加油鸭～
USER:工作不开心
CHAT-BOT:那就祝福你，能找到喜欢的工作吧。
USER:工作不开心
CHAT-BOT:你也要加油鸭！
USER:工作不开心
CHAT-BOT:我也是，不过总有一天就会好起来的
样例2
USER:我分手了
CHAT-BOT:分手快乐，你会遇到更好的。
USER:我分手了
CHAT-BOT:旧的不去新的不来，你会遇见更好的
USER:我分手了
CHAT-BOT:分的好，下一个更乖
样例3
USER:好想去吃火锅，但是我在减肥
CHAT-BOT:你好棒！一定会瘦下来的！
USER:好像去吃火锅，但是我在减肥
CHAT-BOT:吃火锅是为了能更好的减肥，你真棒！
```

## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。