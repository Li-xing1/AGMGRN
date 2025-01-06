# Adaptive Gated Meta Graph Retention Network : A model for urban traffic flow prediction

This is a PyTorch implementation of the paper [Adaptive Gated Meta Graph Retention Network : A model for urban traffic flow prediction]

If you use this code for your research, please cite:
```
@article{ye2021meta,
  title = {Meta Graph Transformer: A Novel Framework for Spatial–Temporal Traffic Prediction},
  journal = {Neurocomputing},
  year = {2021},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2021.12.033},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231221018725},
  author = {Xue Ye and Shen Fang and Fang Sun and Chunxia Zhang and Shiming Xiang},
  publisher={Elsevier}
}
```

## Train

- Check `requirements.txt`
- Unzip `data.zip`
- Train:
  ```shell
  python main.py --model_name <model_name> --dataset_name <dataset_name> --epochs --batch_size <batch_size>
  ```
  For example, 
  ```shell
  python main.py --model_name AGMGRN --dataset_name WHBT --epochs 10 --batch_size 16 
  ```
  means training AGMGRN model for dataset WHBT, the epochs is set to 10 and the batch_size is set to 16.
