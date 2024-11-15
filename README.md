## Preparation
### Prerequisites
Tested with
* PyTorch 1.4
* CUDA 10.0
* Python 3.8
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).

We advise to create a new conda environment for installation. 

PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

```
$ cd xmuda
$ pip install -ve .
```
The `-e` option means that you can edit the code on the fly.

### Datasets
#### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org) and extract it.

You need to perform preprocessing to generate the data for SSE-xMUDA.

Please edit the script `xmuda/data/nuscenes/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the NuScenes dataset
* `out_dir` should point to the desired output directory to store the pickle files
#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org) and additionally the [color data](https://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). 

Extract everything into the same folder.

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well as the segmentation labels to a pickle file.

Please edit the script``` xmuda/data/semantic_kitti/preprocess.py ```as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files
#### A2D2
Please download the Semantic Segmentation dataset and Sensor Configuration from the [Audi website](https://www.a2d2.audi/a2d2/en/download.html), then extract.

For preprocessing, we undistort the images and store them separately as .png files. 

Similar to NuScenes preprocessing, we save all points that project into the front camera image as well as the segmentation labels to a pickle file.

Please edit the script``` xmuda/data/a2d2/preprocess.py ```as follows and then run it.
* `root_dir` should point to the root directory of the A2D2 dataset
* `out_dir` should point to the desired output directory to store the undistorted images and pickle files. It should be set differently than the``` root_dir ```to prevent overwriting of images.
## Training on AS-to-N
You can run the training with
```
$ python xmuda/train_BEV_DG.py --cfg=configs/AS_to_N/xmuda.yaml 
```

## Testing
You can provide which checkpoints you want to use for testing. We used the ones
that performed best on the validation set during training (the best val iteration for 2D and 3D is
shown at the end of each training). Note that `@` will be replaced
by the output directory for that config file. For example:
```
$ python xmuda/test.py --cfg=configs/AS_to_N/xmuda.yaml  @/model_2d_065000.pth @/model_3d_095000.pth
```
You can also provide an absolute path without `@`. 

    
## Acknowledgment
Note that this code is built based on [xMUDA](https://github.com/valeoai/xmuda).
