<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

# Diagnostic Spatio-temporal Transformer with Faithful Encoding

This repository is the official implementation of "Diagnostic Spatio-temporal Transformer with Faithful Encoding". 

![plot](./DFStrans.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```


## Pre-trained Models

You can download pretrained models here:


## Results

Our model achieves the following performance on :


|     Model name     |    Precision    |     Recall     |     F1-Score     |
| ------------------ |---------------- | -------------- | ---------------- | 
| DFStrans           |      0.979      |     0.955      |                  |
| Strans             |      0.977      |     0.917      |                  | 
| MultiHead1DCNN     |      0.993      |     0.914      |                  | 
| InceptionTime      |        1        |     0.892      |                  |
| TapNet             |      0.398      |     0.711      |                  |
| MLSTM-FCN          |      0.913      |                |                  |



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
