# Multi-feature-fuson-fusion-NER（MUFF）
### This project proposes a fused multi-feature NER model for Chinese named entity recognition.
# 1. Code Description
## 1.1 Operational environment
Python3.6~3.8  
transformers=3.0.0

##  1.2 Pre-trained models that need to be downloaded in advance
Pre-trained embeddings (word embeddings, character embeddings and double character embeddings) are the same as Lattice LSTM (only Word(Lattice) embeddings are used in this project)
[下载地址](https://github.com/jiesutd/LatticeLSTM)

### Introduction to Word(Lattice) embeddings: 
Word(Lattice) embeddings(Just download it directly and place it in the data directory)：[ctb.50d.vec](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing)
## 1.3 代码运行
##### Train the .all file in the Weibo dataset and keep the results in the result file:
``!python main.py --resultfile="result/demo.txt"``

##### （The code for training other datasets is similar）
