# Physical-Virtual Collaboration Modeling for Intra-and Inter-Station Metro Ridership Prediction
This is a PyTorch implementation of **Physical-Virtual Collaboration Modeling for Intra-and Inter-Station Metro Ridership Prediction**. 

Due to the widespread applications in real-world scenarios, metro ridership prediction is a crucial but challenging task in intelligent transportation systems. However, conventional methods either ignore the topological information of metro systems or directly learn on physical topology, and cannot fully explore the patterns of ridership evolution. To address this problem, we model a metro system as graphs with various topologies and propose a unified Physical-Virtual Collaboration Graph Network (PVCGN), which can effectively learn the complex ridership patterns from the tailor-designed graphs. Specifically, a physical graph is directly built based on the realistic topology of the studied metro system, while a similarity graph and a correlation graph are built with virtual topologies under the guidance of the inter-station passenger flow similarity and correlation. These complementary graphs are incorporated into a Graph Convolution Gated Recurrent Unit (GC-GRU) for spatial-temporal representation learning. Further, a Fully-Connected Gated Recurrent Unit (FC-GRU) is also applied to capture the global evolution tendency. Finally, we develop a Seq2Seq model with GC-GRU and FC-GRU to forecast the future metro ridership sequentially. Extensive experiments on two large-scale benchmarks (e.g., Shanghai Metro and Hangzhou Metro) well demonstrate the superiority of our PVCGN for station-level metro ridership prediction. \Revise{Moreover, we apply the proposed PVCGN to address the online origin-destination (OD) ridership prediction and the experiment results show the universality of our method.


If you use this code for your research, please cite our papers [https://arxiv.org/abs/2001.04889](https://arxiv.org/abs/2001.04889) :

```
@article{liu2020physicalvirtual,
  title={Physical-Virtual Collaboration Modeling for Intra-and Inter-Station Metro Ridership Prediction},
  author={Lingbo Liu and Jingwen Chen and Hefeng Wu and Jiajie Zhen and Guanbin Li and Liang Lin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2020},
  publisher={IEEE}
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

