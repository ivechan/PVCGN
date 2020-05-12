
### code for "Physical-Virtual Collaboration Graph Network for Station-Level
Metro Ridership Prediction
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
data/model/ggnn_hz_multigraph_rnn256_global_local_fusion_input.yaml
```

- HZMetro
```
python ggnn_train.py --config
data/model/ggnn_hz_multigraph_rnn256_global_local_fusion_input.yaml
```
#### 4. test
[TODO] we would upload trained model later.

