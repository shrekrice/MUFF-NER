# Multi-feature-NER
### 该项目提出一种融合多特征的NER模型用于中文命名实体识别。
# 1.代码描述
## 1.1 运行环境
Python3.6~3.8  
transformers=3.0.0
##  1.2 需要提前下载的预训练模型
预训练嵌入（词嵌入、字符嵌入和双字符嵌入）与 Lattice LSTM 相同（本项目只使用Word(Lattice) embeddings）
[下载地址](https://github.com/jiesutd/LatticeLSTM)
### 文件名介绍：
主要使用Word(Lattice) embeddings(直接下载即可，并将其置放在data目录下)：[ctb.50d.vec](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing)
## 1.3 代码运行
##### 训练Weibo数据集中的.all文件，并且将结果保留在result文件中：
``!python main.py --train data/WeiboNER/train.all.bmes --dev data/WeiboNER/dev.all.bmes --test data/WeiboNER/test.all.bmes --modelname Weibo --savedset data/Weibo.dset --lr=0.005 --hidden_dim 200 --num_iter=30 --resultfile="result/demo.txt"``
##### （训练其他数据集代码类似）
