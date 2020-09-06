# PerMo

## Network & 3D reconstruction
### Requirements
* Python ≥ 3.6, PyTorch ≥ 1.4
* opencv, tqdm
* [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

### Usage
Step 1. Get the part segmentation, uv regression using our [pre-trained model](https://drive.google.com/file/d/1qsuVn1J4E3XJhrj9ijfjgm_1H1TToaM2/view?usp=sharing).
```
python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml kitti3d.pth [path to images] --output stage_1_res.pkl -v
```
To vis stage_1_res.pkl, modify config.yaml to set input(input_image_dir, stage1_network_res) and ouput path.
```
python vis_pkl.py 
```
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_part.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_u.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_v.png" width="860"/>

Step 2. Sovle pose and reconstruct vehicle models from [Step 1's result](https://drive.google.com/file/d/1-3phQ23taaeO3mpo3z0DNAuMs60d40mI/view?usp=sharing).
Download the [template_models](https://drive.google.com/file/d/10o8a_TQo3633ArHikg0Pgkzb-ZJNfw-e/view?usp=sharing), [simplication_template_models](https://drive.google.com/file/d/1FC685JatxTlHmRwtnItfEkSZLWs926Ut/view?usp=sharing), [camera calib](https://drive.google.com/file/d/1VmX_S3jCYnfuj8CLKuv6X2x1tZ5IiB6q/view?usp=sharing), [skp pre-trained model](https://drive.google.com/file/d/1H5Quk4s8kq2BEZLBBahmwRbetxu0qUO8/view?usp=sharing) and [estimated depth map](https://drive.google.com/file/d/1LVsyKJ4PLVMk-ECqQkMqE35ejPn0pBwS/view?usp=sharing). Modify config.yaml to set resource and ouput path.
```
python solve.py
```
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047.png" width="860"/>

## Dataset
We use 28 industrial grade vehicle CAD models(including five vehicle classes: coupe, hatchback, notchback, SUV, MPV) to label and fit KITTI training dataset which contains 6871 images and 33747 car instances in total. We can generate instance segmentation, part-level segmentation and uv coordinates. Part of our annotations can be downloaded at [here](https://drive.google.com/file/d/1zKTJbnANpIdLA3MXNz_ePHWYNXuVzPET/view?usp=sharing).


![Kitti labeled example](https://github.com/SA2020PerMo/PerMo/blob/master/3D_Tool/vis/006127.png)

## Labelling Tool

### Description

* This tool can be used to label the 6DOF pose and type of the vehicles in images. 
* We have successfully used this tool on Kitti and Apollo.




### Requirements

* Ubuntu/MacOS/Windows
* Python3
* PyQt5
* opencv-python
* numpy

### Usage

    python win.py
* Slide the x, y, z, a, b and c to change the pose of the car.
* Choose car's type.
* The raw images are under /images.
* The label results are under /label_result.
* The camera information are under /calib.
* We provide 28 car models, which are under /models.
![](https://github.com/SA2020PerMo/PerMo/blob/master/3D_Tool/vis/tool2.png)




