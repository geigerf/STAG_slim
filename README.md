# Learning the signatures of the human grasp using a scalable tactile glove

## Introduction
This is an adaptation of the Pytorch based code for object classification and object estimation methods
presented in the paper "Learning the signatures of the human grasp using a scalable tactile glove".
It uses only the classification newtork, not the weight estimation network and the training/test loss
evolution over number of epochs is saved.

It relies on Pytorch 0.4.1 (or newer) and the dataset that can be downloaded separately from
[http://stag.csail.mit.edu/#data](http://stag.csail.mit.edu/#data) .


## System requirements

Requires CUDA and Python 3.6+ with following packages (exact version may not be necessary):

* numpy (1.15.4)
* torch (0.4.1)
* torchfile (0.1.0)
* torchvision (0.2.1)
* scipy (1.1.0)
* scikit-learn (0.19.1)

## Dataset preparation

1. Download the `classification` dataset from
[http://stag.csail.mit.edu/#data](http://stag.csail.mit.edu/#data) .
2. Extract the dataset metadata.mat files to a sub-folder `data\[task]`.
The resulting structure should be something like this:
```
data
|--classification
|    |--metadata.mat
```
Use a runtime argument
`--dataset [path to metadata.mat]` to specify its location.
The images in the dataset are for illustration only and are not used by this code.
More information about the dataset structure is availble in [http://stag.csail.mit.edu](http://stag.csail.mit.edu).

## Object classification

Run the code from the root working directory (the one containing this readme).

### Training
You can train a model from scratch for `N` input frames using:
```
python classification/main.py --reset --dataset [path to metadata.mat] --nframes N
```
You can change the location of the saved snapshots using `--snapshotDir YOUR_PATH`.


## History

* **March 2020**: Code uploaded to GitLab.


## Contact

Please email any questions or comments to [geigerf@student.ethz.ch](geigerf@student.ethz.ch).
