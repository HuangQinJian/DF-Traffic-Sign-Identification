# Baseline

**一、安装**

地址：[MaskRCNN-Benchmark(Pytorch版本)](https://github.com/facebookresearch/maskrcnn-benchmark)

首先要阅读官网说明的[环境要求](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)，**千万不要一股脑直接安装，不然后面程序很有可能会报错！！！** 

> 
> - PyTorch 1.0 from a nightly release. It will not work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
>- torchvision from master
>- cocoapi
>- yacs
>- matplotlib
>- GCC >= 4.9
>- OpenCV

```
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
一定要按上面的说明一步一步来，千万别省略，**不然后面程序很有可能会报错！！！** 

**二、训练**

把数据处理好就可以跑起来了。
