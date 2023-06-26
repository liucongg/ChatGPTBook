本项目为书籍《ChatGPT原理与实战：大型语言模型的算法、技术和私有化》中第9章《类ChatGPT实战》实战部分代码-SFT阶段。

## 项目简介

SFT 阶段实战，通过文档生成问题任务，更加深入理解ChatGPT模型在SFT阶段的任务流程。

项目主要结构如下：

- data 存放数据的文件夹
    - cmrc_train.json 原始机器阅读理解训练数据
    - cmrc_dev.json 原始机器阅读理解测试数据
- sft_model 已训练好的模型路径
    - config.json
    - pytorch_model.bin
    - vocab.txt
- pretrain_model 预训练文件路径
    - config.json
    - pytorch_model.bin
    - vocab.txt
- data_helper.py 数据预处理文件
- data_set.py 模型所需数据类文件
- model.py 模型文件
- train.py 模型训练文件
- predict.py 模型推理文件

注意：由于GitHub不方便放模型文件，因此sft_model文件夹和pretrain_model文件夹中的模型bin文件，请从百度云盘中下载。

| 文件名称 | 下载地址 | 提取码 |
| --- |--- |---|
| pretrain_model | [百度云](https://pan.baidu.com/s/11cBXuwhBFQidelrW-lTWnw) | 9nlh|
| sft_model |[百度云](https://pan.baidu.com/s/17HqIbhklJwAv48lDbIZDjg) |dn8d|

## 环境配置

模型训练或推理所需环境，请参考[requirements.txt](../requirements.txt)文件。

## 数据处理

数据预处理需要运行data_helper.py文件，会在data文件夹中生成训练集和测试集文件。

命令如下：

```shell
python3 data_helper.py
```

注意：如果需要修改数据生成路径或名称，请修改data_helper.py文件67-71行，自行定义。

## 模型训练

模型训练需要运行train.py文件，会自动生成output_dir文件夹，存放每个epoch保存的模型文件。

命令如下：

```shell
python3 train.py --device 0 \
                 --data_dir "data/" \
                 --train_file_path "data/sft_train.json" \
                 --test_file_path "data/test.json" \
                 --pretrained_model_path "pretrain_model/" \
                 --max_len 768 \
                 --query_max_len 64 \
                 --train_batch_size 8 \
                 --test_batch_size 8 \
                 --num_train_epochs 5  
```

注意：当服务器资源不同或读者更换数据等时，可以在模型训练时修改响应参数，详细参数说明见代码或阅读书9.3.1小节。

模型训练示例如下：

![img.png](../images/9_1.png)

模型训练阶段损失值变化如下：

![img.png](../images/9_2.png)

## 模型推理

模型训练需要运行predict.py文件，可以采用项目中以提供的模型，也可以采用自己训练后的模型。

命令如下：

```shell
python3 predict.py --device 0 --model_path sft_model
```

注意：如果修改模型路径，请修改--model_path参数。

模型推理示例如下：

![img.png](../images/9_3.png)

```text
样例1：
输入的正文为：大莱龙铁路位于山东省北部环渤海地区，西起位于益羊铁路的潍坊大家洼车站，向东经海化、寿光、寒亭、昌邑、平度、莱州、招远、终到龙口，连接山东半岛羊角沟、潍坊、莱州、龙口四个港口，全长175公里，工程建设概算总投资11.42亿元。铁路西与德大铁路、黄大铁路在大家洼站接轨，东与龙烟铁路相连。
生成的第1个问题为：大莱龙铁路位于哪些地方？
生成的第2个问题为：为什么该线是由莱芜水泥厂承包？
样例2：
输入的正文为：椰子猫（学名：'），又名椰子狸，为分布于南亚及东南亚的一种麝猫。椰子猫平均重3.2公斤，体长53厘米，尾巴长48厘米。它们的毛粗糙，一般呈灰色，脚、耳朵及吻都是黑色的。它们的身体上有三间黑色斑纹，面部的斑纹则像浣熊，尾巴没有斑纹。椰子猫是夜间活动及杂食性的。它们在亚洲的生态位与在北美洲的浣熊相近。牠们栖息在森林、有树木的公园及花园之内。它们的爪锋利，可以用来攀爬。椰子猫尾巴下有嗅腺，形状像睾丸，可以分泌有害的物质。椰子猫分布在印度南部、斯里兰卡、东南亚及中国南部。
生成的第1个问题为：椰子猫是什么样的品种？
生成的第2个问题为：椰子猫现在在哪些地方生活？
```

## 总结

本项目中的代码包含大量的注释信息，帮助读者更容易的阅读代码、以及了解其原理。读者跑通代码的后，可以根据自己特定的任务，定向修改配置参数或代码，实现自己响应的功能。