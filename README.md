## Autopilot-TensorFlow
A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes.

## Source of Data
 - NVidia dataset: 72 hrs of video => 72*60*60*30 = 7,776,000 images
 - Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
 - Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]
 - More data: https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5

##  Source / Useful links
 - https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5
 - https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/
 - https://www.youtube.com/watch?v=qhUvQiKec2U
 - https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c

##  Prerequisites
You need to have installed following softwares and libraries before running this project.
1. Python 3: https://www.python.org/downloads/
2. Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy and scipy: https://www.anaconda.com/download/

## How to Use
Download the [dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing) and extract into the repository folder

Use `python train.py` to train the model

Use `python run.py` to run the model on a live webcam feed

Use `python run_dataset.py` to run the model on the dataset

To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.


## Libraries
* __tensorflow:__ TensorFlow provides multiple APIs.The lowest level API, TensorFlow Core provides you with complete programming control.
    * pip install tensorflow
    * conda install -c anaconda tensorflow

* __opencv:__ OpenCV-Python is the Python API of OpenCV. It combines the best qualities of OpenCV C++ API and Python language.
    * conda install -c conda-forge opencv
    

#### Credits: https://github.com/SullyChen/Autopilot-TensorFlow
