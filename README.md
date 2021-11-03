**Arch-Net**: Model Distillation for Architecture Agnostic Model Deployment
========
The official implementation of [Arch-Net: Model Distillation for Architecture Agnostic Model Deployment](https://arxiv.org/abs/2111.01135)

## Introduction
**TL;DR** Arch-Net is a family of neural networks made up of simple and efficient operators. When a Arch-Net is produced, less common network constructs, like Layer Normalization and Embedding Layers, are eliminated in a progressive manner through label-free Blockwise Model Distillation, while performing sub-eight bit quantization at the same time to maximize performance. For the classification task, only 30k unlabeled images randomly sampled from ImageNet dataset is needed.

## Main Results

ImageNet Classification

|  **Model**  | **Bit Width** | **Top1** | **Top5**  |
|:------:|:------:|:------:|:------:|
|Arch-Net_Resnet18|32w32a|69.76|89.08|
|Arch-Net_Resnet18|2w4a|**68.77**|88.66|
|Arch-Net_Resnet34|32w32a|73.30|91.42|
|Arch-Net_Resnet34|2w4a|**72.40**|91.01|
|Arch-Net_Resnet50|32w32a|76.13|92.86|
|Arch-Net_Resnet50|2w4a|**74.56**|92.39|
|Arch-Net_MobilenetV1|32w32a|68.79|88.68|
|Arch-Net_MobilenetV1|2w4a|**67.29**|88.07|
|Arch-Net_MobilenetV2|32w32a|71.88|90.29|
|Arch-Net_MobilenetV2|2w4a|**69.09**|89.13|

Multi30k Machine Translation

|  **Model**  | **translation direction** | **Bit Width**  | **BLEU**  |
|:------:|:------:|:------:|:------:|
|Transformer|English to Gemany|32w32a|32.44|
|Transformer|English to Gemany|2w4a|**33.75**|
|Transformer|English to Gemany|4w4a|34.35|
|Transformer|English to Gemany|8w8a|36.44|
|Transformer|Gemany to English|32w32a|30.32|
|Transformer|Gemany to English|2w4a|**32.50**|
|Transformer|Gemany to English|4w4a|34.34|
|Transformer|Gemany to English|8w8a|34.05|

## Dependencies
python == 3.6

refer to requirements.txt for more details

## Data Preparation
Download ImageNet and multi30k data([google drive](https://drive.google.com/file/d/1yVa7C41Xpz8kAt_YlNT5xz99IshDE5XT/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1OJyh--BiP3n52GzeINV2tQ), code: 8brd) and put them in ./arch-net/data/ as follow:

```
./data/
├── imagenet
│   ├── train
│   ├── val
├── multi30k
```

Download teacher models at [google drive](https://drive.google.com/file/d/1vnDsVboTrjbG9DgUvmWqyZv1rBv7uKbC/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1UA1GJXEzM-S6rcwEyl4oMw)(code: 57ew) and put them in ./arch-net/models/teacher/pretrained_models/

## Get Started

### ImageNet Classification (take archnet_resnet18 as an example)
```sh
cd ./train_imagenet
```

train and evaluate
```sh
python3 -m torch.distributed.launch --nproc_per_node=8 train_archnet_resnet18.py  -j 8 --weight-bit 2 --feature-bit 4 --lr 0.001 --num_gpus 8 --sync-bn
```

evaluate if you already have the trained models
```sh
python3 -m torch.distributed.launch --nproc_per_node=8 train_archnet_resnet18.py  -j 8 --weight-bit 2 --feature-bit 4 --lr 0.001 --num_gpus 8 --sync-bn --evaluate
```

### Machine Translation

```sh
cd ./train_transformer
```

train a arch-net_transformer of 2w4a

```sh
python3 train_archnet_transformer.py --translate_direction en2de --batch_size 48 --final_epochs 50 --weight_bit 2 --feature_bit 4 --label_smoothing
```

evaluate

```sh
python3 translate.py --data_pkl ./data/multi30k/m30k_ende_shr.pkl --model path_to_the_outptu_directory/model_max_acc.chkpt
```

- to get the BLEU of the evaluated results, go to [this website](https://www.letsmt.eu/Bleu.aspx), and then upload 'predictions.txt' in the output directory and the 'gt_en.txt' or 'gt_de.txt' in ./arch-net/data_gt/multi30k/

## Citation

If you find this project useful for your research, please consider citing the paper.
```
@misc{xu2021archnet,
      title={Arch-Net: Model Distillation for Architecture Agnostic Model Deployment}, 
      author={Weixin Xu and Zipeng Feng and Shuangkang Fang and Song Yuan and Yi Yang and Shuchang Zhou},
      year={2021},
      eprint={2111.01135},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
[attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

[LSQuantization](https://github.com/hustzxd/LSQuantization)

[pytorch-mobilenet-v1](https://github.com/wjc852456/pytorch-mobilenet-v1)

## Contact
If you have any questions, feel free to open an issue or contact us at xuweixin02@megvii.com.
