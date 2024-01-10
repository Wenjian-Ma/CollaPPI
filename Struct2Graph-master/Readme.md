Baseline
---
We retrained the baseline method, Struct2Graph, with its default parameters using the yeast dataset in our paper.

1. The preprocessed data in yeast dataset used for Struct2Graph can be available at the [Link](https://pan.baidu.com/s/1mrJ5HQ2wMp1Wv0D3YI72Cg?pwd=1234).

2. The trained model on yeast dataset can be available at the [Link](https://pan.baidu.com/s/19KpAuXthWU6RZTF5FORPhA?pwd=1234), which is used to reproduce the performance of Struct2Graph recorded in our paper.

    Under the path of Struct2Graph-master/ :  `python test.py`

3. If you want to retrain the Struct2Graph on the yeast dataset in our paper:

    Under the path of Struct2Graph-master/ :  `python k-fold-CV.py`
