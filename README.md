# CollaPPI
Code for paper "CollaPPI: A Collaborative Learning Framework for Predicting Protein-Protein Interactions"
---

Dependencies
---

python == 3.7.13

pytorch == 1.7.1

PyG (torch-geometric) == 2.0.4

torch-cluster == 1.5.9

torch-scatter == 2.0.5

torch-sparse == 0.6.8

torch-spline-conv == 1.2.0

sklearn == 1.0.2

scipy == 1.7.3

numpy == 1.21.5

Data preparation
---
1. The relevant data and trained model of _yeast_ (~6.14G) can be available at the [Link](https://pan.baidu.com/s/1kknFC2gpayvxLM_1sqwO7w?pwd=1234).

2. The relevant data and trained model of _multi-species_ (~24.16G) can be available at the [Link](https://pan.baidu.com/s/1kQHXCAQxzNO5peLqJni8xg?pwd=1234).

3. The relevant data and trained model of _multi-class_ (~11.60G) can be available at the [Link](https://pan.baidu.com/s/18VNZJzcRQCN8myJ8Pb6SAA?pwd=1234).

4. Unzip the above file to the corresponding directory (e.g., dictionary_yeast.tar.gz should be extracted to `./data/yeast`).

5. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

Train
---
`python main.py`

Test
---
`python test.py` used to reproduct the performence recorded in the paper.

`python test_mul.py` multiplication for mutual interaction. Tranined model can be available at the [Link](https://pan.baidu.com/s/1QgK3w80w08U_Ywl3aBwc3w?pwd=1234), which is evaluated on yeast dataset.

`python test_tran_mul.py` Transposed multiplication for mutual interaction. Trained model can be available at the [Link](https://pan.baidu.com/s/1E_t8KWFyZfQvo9qCxb25Ag?pwd=1234), which is evaluated on yeast dataset.

Baseline
---

TAGPPI: [https://github.com/xzenglab/TAGPPI](https://github.com/xzenglab/TAGPPI).

PIPR: [https://github.com/muhaochen/seq_ppi](https://github.com/muhaochen/seq_ppi).

Struct2Graph: [https://github.com/baranwa2/Struct2Graph](https://github.com/baranwa2/Struct2Graph).

DPPI: [https://github.com/hashemifar/DPPI/](https://github.com/hashemifar/DPPI/).


We retrained the baseline method, Struct2Graph, with its default parameters using the yeast dataset in our paper.

1. The preprocessed data in yeast dataset used for Struct2Graph can be available at the [Link](https://pan.baidu.com/s/1mrJ5HQ2wMp1Wv0D3YI72Cg?pwd=1234).

2. The trained model on yeast dataset can be available at the [Link](https://pan.baidu.com/s/19KpAuXthWU6RZTF5FORPhA?pwd=1234), which is used to reproduce the performance of Struct2Graph recorded in our paper.

    Under the path of Struct2Graph-master/ :  `python test.py`

3. If you want to retrain the Struct2Graph on the yeast dataset in our paper:

    Under the path of Struct2Graph-master/ :  `python k-fold-CV.py`

Cite our work
---
if you use the conclusion, code, or data in our work, please cite:
```
@ARTICLE{10465250,
  author={Ma, Wenjian and Bi, Xiangpeng and Jiang, Huasen and Zhang, Shugang and Wei, Zhiqiang},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={CollaPPI: A Collaborative Learning Framework for Predicting Protein-Protein Interactions}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Proteins;Collaboration;Task analysis;Feature extraction;Protein engineering;Deep learning;Vectors;protein-protein interaction;multi-task learning;graph neural network;protein representation learning},
  doi={10.1109/JBHI.2024.3375621}}
```
