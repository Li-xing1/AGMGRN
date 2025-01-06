# Adaptive Gated Meta Graph Retention Network : A model for urban traffic flow prediction

This is a PyTorch implementation of the paper [Adaptive Gated Meta Graph Retention Network : A model for urban traffic flow prediction]

If you use this code for your research, please cite:
```

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
