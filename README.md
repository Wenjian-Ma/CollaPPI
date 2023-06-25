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
`python test.py`
