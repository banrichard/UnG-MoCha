UnG-MoCha:Neural Motif Counting in Uncertain Graphs
-----------------
A PyTorch + torch-geometric implementation of UnG-MoCha.
### Requirements
```
matplotlib==3.5.2
networkx==2.8.4
numpy==1.21.5
pandas==1.4.4
scikit_learn==1.0.2
scipy==1.9.1
torch==2.0.0
torch_geometric==2.4.0
torch_scatter==2.1.2
torch_sparse==0.6.18
tqdm==4.64.1
```
Install the required packages by running
```
pip install -r requirements.txt
```
### Quick Start
Model Training and Evaluation
```
python train.py --graph_net GINE --motif_net NNGINConcat --predict_net CCANet --queryset_dir queryset \
    --epochs 200 --batch_size 16 --lr 0.0006 --weight_decay 0.0005 --weight_decay_var  0.1 --decay_factor 0.7\
    --true_card_dir label --dataset krogan --data_dir dataset --dataset_name krogan_core.txt --save_res_dir result \
    --save_model_dir saved_model
```
Model Evaluation
```
python train.py --test_only True
```
Running UnG-MoCha without USSL
```
python train.py --graph_net GINE --motif_net NNGINConcat --predict_net CCANet --queryset_dir queryset \
    --epochs 200 --batch_size 16 --lr 0.0006 --weight_decay 0.0005 --weight_decay_var  0.1 --decay_factor 0.7\
    --true_card_dir label --dataset krogan --data_dir dataset --dataset_name krogan_core.txt --save_res_dir result \
    --save_model_dir saved_model --gsl False
```
Running case visualization (USSL based)
```
python train.py --graph_net GINE --motif_net NNGINConcat --predict_net CCANet --queryset_dir queryset \
    --epochs 200 --batch_size 16 --lr 0.0006 --weight_decay 0.0005 --weight_decay_var  0.1 --decay_factor 0.7\
    --true_card_dir label --dataset krogan --data_dir dataset --dataset_name krogan_core.txt --save_res_dir result \
    --save_model_dir saved_model --visualization_only True
```
Running parallel train (We recommend to use 4 A100 NVLink or better (H100) GPUs to achieve the best performance.)
```
 torchrun --standalone --nnodes=1 --nproc_per_node=4 ./para_train.py --dataset krogan --dataset_name krogan_core.txt --graph_net GraphSage

```


### Key Parameters
All the parameters with their default value are in active_train.py

| name             | type   | description                                                     | 
|------------------|--------|-----------------------------------------------------------------|
| graph_net        | String | type of component GNN layer (GIN, GINE, GAT, GCN, SAGE)         |
| motif_net        | String | type of motif NN layer (GIN, GINE, GAT, GCN, SAGE, NNGINConcat) |
| num_layers       | Int    | number of GNN layers                                            |
| epochs           | Int    | number of training epochs                                       |
| batch_size       | Int    | mini-batch size for sgd                                         |
| k                | Int    | number of hops to extract ego nets                              |
| lr               | Float  | learning rate                                                   |
| weight_decay_var | Float  | Trade-off parameter for predicted mean and variance             |
| decay_factor     | Float  | Decay factor for ExponentialLR scheduler                        |
