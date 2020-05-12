
### code for "Physical-Virtual Collaboration Graph Network for Station-Level Metro Ridership Prediction
#### 1. requirements
- python3
- numpy
- yaml
- pytorch
- torch_geometric
#### 2. extract  datasets
```
cd data && tar xvf data.tar.gz
```
#### 3. train
- SHMetro
```
python ggnn_train.py --config
data/model/ggnn_sh_multigraph_rnn256_global_local_fusion_input.yaml
```

- HZMetro
```
python ggnn_train.py --config
data/model/ggnn_hz_multigraph_rnn256_global_local_fusion_input.yaml
```
#### 4. test
First of all, download the trained model and extract it to `trained` folder.

- [download from dropbox ](https://www.dropbox.com/s/37ep6jafampcavf/trained.zip?dl=0)
- [百度网盘下载，提取码：np5p](https://pan.baidu.com/s/1lesAk4WOfBQtg0a0XgDfvA)

 ```
mkdir trained
mv trained.zip trained
cd trained && unzip trained.zip
```
- SHMetro
```
python ggnn_evaluation.py --config trained/sh.yaml
```
- HZMetro
```
python ggnn_evaluation.py --config trained/hz.yaml
```

