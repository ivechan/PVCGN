## SHMetro & HZMetro Dataset

In this work, we focus on the metro ridership prediction from 5:30 - 23:30. Specifically, we utilize the metro ridership (inflow/outflow) of the previous four time intervals (4x15m=60m) to predict the metro ridership (inflow/outflow) of future four time intervals (4x15m=60m) ::
```
5:30-6:30 -- forecast -> 6:30-7:30
5:45-6:45 -- forecast -> 6:45-7:45
...
21:15-22:15 -- forecast -> 22:15-23:15
21:30-22:30 -- forecast -> 22:30-23:30
```
Therefore, each day can be split into 66 time slices. For each dataset, we release six ```pkl``` files, three for metro ridership data, and three for metro graph information.

### 1. Metro Ridership
In our work, each dataset is divided into a training set, a validation set, and a testing set.
* **train.pkl**: the training set. It is a ```dict``` that consists of 4 ```ndarray```:
```
(1) x: the metro ridership (inflow/outflow) of the previous four time intervals. Its shape is [T, n, N, D]. 
(2) y: the metro ridership (inflow/outflow) of the next four time intervals. Its shape is [T, m, N, D]. 
(3) xtime: the timestamps of x. Its shape is [T, n]. 
(4) ytime: the timestamps of y. Its shape is [T, m].

T = the number of time slices
N = the number of metro stations
n = the length of the input sequence,  i.e, 4 time intervals in our work
m = the length of the output sequence, i.e, 4 time intervals in our work
D = the data dimension of each station, i.e, 2 (inflow/outflow) in our work
```

* **val.pkl**: the validation set. Its data organization is similar to that of the training set.
* **test.pkl**: the testing set.   Its data organization is similar to that of the training set.


For each time slice, we use ```x[i]``` to predict ```y[i]```.
In the SHMetro dataset, the ridership data of 62 days is used for training, thus the shape[0] of ```x``` is 62x66=4092.

In our work, sequence lengths ```n``` and ```m``` are uniformly set to 4. You can also adopt other lengths by reorganizing our data.

### 2.Graph Information
* **graph_conn.pkl**: the physical graph of metro
* **graph_sml.pkl**: the similarity graph of metro
* **graph_conn.pkl**: the correlation graph of metro
