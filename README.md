# Learning the signatures of the human grasp using a scalable tactile glove

## Introduction
This is an adaptation of the Pytorch based code for object classification and object estimation methods
presented in the paper _Subramanian Sundaram, Petr Kellnhofer, Yunzhu Li, Jun-Yan Zhu, Antonio Torralba and Wojciech Matusik. “Learning the signatures of the human grasp using a scalable tactile glove”. Nature, 2019_.
It uses only the classification network, not the weight estimation network and the training/test loss
evolution over number of epochs is saved.

It relies on Pytorch 0.4.1 (or newer) and the dataset that can be downloaded separately from
[http://stag.csail.mit.edu/#data](http://stag.csail.mit.edu/#data) .


## System requirements

This lists the installed python packages and versions.
However, it is not absolutely necessary to install it with exactly the same versions.
Consult this list if there are some broken dependencies in your environment.

Required packages:
- Python            3.8.1
- numpy             1.18.1
- pytorch           1.4.0 CUDA version
- imbalanced-learn  0.6.2
- scikit-learn      0.22.1
- scipy             1.4.1

Imported standard packages:
- argparse
- collections
- datetime
- os
- random
- re
- shutil
- sys
- time

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
