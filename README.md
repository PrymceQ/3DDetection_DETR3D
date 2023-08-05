# 3DDetection_DETR3D
Private The reproduction project of the 3D Detection model # DETR3D, which includes some code annotation work

Thanks for the BEVFusion authors！[Paper](https://arxiv.org/abs/2110.06922) | [Code](https://github.com/WangYueFt/detr3d)

## 🌵Necessary File Format
- mmdetection3d/ # https://github.com/open-mmlab/mmdetection3d/
- data/nuscenes/
  - maps/
  - samples/
  - sweeps/
  - v1.0-test/
  - v1.0-trainval/
- pretrained/
- projects/
  - configs/
  - mmdet3d_plugin/
- tools/
- work_dirs/detr3d_res101_gridmask/

## 🌵Build Envs
You can refer to the official configuration environment documentation. [Official Git](https://github.com/WangYueFt/detr3d)
> BEVFusion's official introduction document is very comprehensive and detailed.

Or use the Conda env configuration file we provide.
```
conda env create -f detr3d_env.yaml
```

And then, you should git clone from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/) to build - mmdetection3d/

## 🌵Data create

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## Pretrained ckpts download

According to [Official Git](https://github.com/WangYueFt/detr3d), you should download pretrained weights from [Pretrained weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN) and set them to - pretrained/

## 🌵Train Code
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 4 
```

Or you can directly use the config we provided.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
tools/dist_train.sh work_dirs/detr3d_res101_gridmask/detr3d_res101_gridmask.py 4
```


## 🌵Test Code

```
tools/dist_test.sh work_dirs/detr3d_res101_gridmask/detr3d_res101_gridmask.py work_dirs/detr3d_res101_gridmask/epoch_24.pth 4 --eval=bbox
```

## 🌵Training Result Record

ID | Name | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Epochs | Data | Batch_size | GPUs | Train_time | Eval_time | Log_file
:----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :-----------
0 | detr3d_res101_gridmask | 0.3496 | 0.3173 | 0.7720 | 0.7075 | 1.5471 | 0.8842 | 0.2119 |  24 | All | 4, sample per gpu=1 | 4 x Nvidia Geforce 3090 | 3days16hours | 235.1s | work_dirs/detr3d_res101_gridmask/

## 🌵Some issues

### 1. TypeError: cannot pickle "dict_keys* oojeet.

![1](https://github.com/PrymceQ/3DDetection_DETR3D/assets/109404970/26b77515-7af1-42d9-8657-435a1b544906)

Set `worker_per_gpu=0` in the config file to solve.

### 2. Long training time analysis.

> The training time takes 3days16hours, which is so long. We analyze the reasons as follows:

- We set `worker_per_gpu=0`. As a result, only one worker is running on each GPU, which may waste the computing power of the GPU, resulting in a decrease in training efficiency.

- Due to the characteristics of the model itself. Since detr3d needs to map the 3d frame back to the position in the 2d picture to get the feature vector, so it is slow.

## 🌵Key Model Files

Here we have made simple annotations on some key model files in Chinese, these annotations are based on "projects/configs/detr3d/detr3d_res101_gridmask.py" config. 

You can find them in:
- projects/mmdet3d_plugin/models/detectors/detr3d.py
- projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py
- projects/mmdet3d_plugin/models/utils/detr3d_transformer.py
