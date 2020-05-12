
# Physical-Virtual Collaboration Graph Network for Station-Level Metro Ridership Prediction
This is a PyTorch implementation of **Physical-Virtual Collaboration Graph Network for Station-Level Metro Ridership Prediction (PVCGN)**. 

Due to the widespread applications in real-world scenarios, metro ridership prediction is a crucial but challenging task in intelligent transportation systems. However, conventional methods that either ignored the topological information of metro systems or directly learned on physical topology, can not fully explore the ridership evolution patterns. To address this problem, we model a metro system as graphs with various topologies and propose a unified Physical-Virtual Collaboration Graph Network (PVCGN), which can effectively learn the complex ridership patterns from the tailor-designed graphs. Specifically, a physical graph is directly built based on the realistic topology of the studied metro system, while a similarity graph and a correlation graph are built with virtual topologies under the guidance of the inter-station passenger flow similarity and correlation. These complementary graphs are incorporated into a Graph Convolution Gated Recurrent Unit (GC-GRU) for spatial-temporal representation learning. Further, a Fully-Connected Gated Recurrent Unit (FC-GRU) is also applied to capture the global evolution tendency. Finally, we develop a seq2seq model with GC-GRU and FC-GRU to forecast the future metro ridership sequentially. Extensive experiments on two large-scale benchmarks (e.g., Shanghai Metro and Hangzhou Metro) well demonstrate the superiority of the proposed PVCGN for station-level metro ridership prediction.


If you use this code for your research, please cite our papers  [https://arxiv.org/abs/2001.04889](https://arxiv.org/abs/2001.04889)):

```
@misc{chen2020physicalvirtual,
    title={Physical-Virtual Collaboration Graph Network for Station-Level Metro Ridership Prediction},
    author={Jingwen Chen and Lingbo Liu and Hefeng Wu and Jiajie Zhen and Guanbin Li and Liang Lin},
    year={2020},
    eprint={2001.04889},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

### Requirements
- python3
- numpy
- yaml
- pytorch
- torch_geometric
### Extract dataset
```
cd data && tar xvf data.tar.gz
```
## Train
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
## Test
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

