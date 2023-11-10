# APTAnet
# An atom-level PTI affinity prediction model

## 1. environment configuration
```
conda env create -f environment.yaml
torch.cuda.is_available()
```


## 2. Code Structure

- The "data" folder contains the data, which is primarily sourced from https://github.com/PaccMann/TITAN and various databases.
- The "experiment" folder includes experiment results files and parameter files.
- "pretrain.py" is the model training code.
- "mydataset.py" is responsible for data loading.
- "APTAnet.py" defines the APTAnet model.
- "knn.py" is the KNN baseline model.
- "plot" contains the code for generating plots.


## 3. Usage

- Activate Conda environment
- check experiment/model_params.json

- In the command-line interface, initiate DDP (Distributed Data Parallel) training, as shown in the example below, using two GPUs:

  ```
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --node_rank=0  pretrain.py
  ```

- The training results can be found in the "experiment" folder.
