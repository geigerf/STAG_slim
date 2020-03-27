#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:37:27 2020

@author: fabian
"""


import scipy.io as sio
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.utils.data as data
import torch


def load_data(filename, augment=False, kfold=3, seed=333):
    """
    Function to load the pressure data with stratified k fold

    Parameters
    ----------
    filename : string
        Path to the data file.
    augment : bool, optional
        Whether or not to add random noise to the pressure data.
        The default is False.
    kFold : int, optional
        Number of folds to use in stratified k fold. The default is 3.
    seed : int, optional
        Seed used for shuffling the train test split and stratified k fold.
        The default is None.

    Returns
    -------
    train_data : numpy array
        DESCRIPTION.
    train_labels : numpy array
        DESCRIPTION.
    train_ind : numpy array
        DESCRIPTION.
    val_ind : numpy array
        DESCRIPTION.
    test_data : numpy array
        DESCRIPTION.
    test_labels : numpy array
        DESCRIPTION.
    """

    data = sio.loadmat(filename)
    valid_idx = np.array(data['hasValidLabel'].flatten()) == 1
    balanced_idx = np.array(data['isBalanced'].flatten()) == 1
    indices = np.logical_and(valid_idx, balanced_idx)
    pressure = np.array(data['pressure'])
    pressure = pressure[indices]
    object_id = np.array(data['objectId']).flatten()
    object_id = object_id[indices]
    # Prepare the data the same way as in the paper
    pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
    # Transform the pressure data into 1D sequences for sklearn utils
    pressure = np.reshape(pressure, (pressure.shape[0], 32*32))

    train_data, test_data,\
        train_labels, test_labels = train_test_split(pressure, object_id,
                                                     test_size=0.3,
                                                     random_state=seed,
                                                     stratify=object_id)
    #print(train_data.shape, train_labels.shape)
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    # train_ind, val_ind = skf.split(train_data, train_labels)
    skf_gen = skf.split(train_data, train_labels)

    if augment:
        noise = np.random.random_sample(train_data.shape)*0.015
        train_data += noise

    return train_data, train_labels, test_data, test_labels, skf_gen


class DataLoader(data.Dataset):
    def __init__(self, data, labels, augment=False, nframes=5,
                 use_clusters=False, nclasses=27):
        self.nframes = nframes
        self.nclasses = nclasses
        self.data = data
        self.labels = labels
        self.use_clusters = use_clusters
        self.collate_data()
        self.dummyrow = torch.zeros((32,1))
        self.dummyimage = torch.zeros((3, 1, 1))


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data = torch.from_numpy(self.collated_data[idx])
        labels = torch.LongTensor([int(self.collated_labels[idx])])
        return self.dummyrow, self.dummyimage, data, labels


    def collate_data(self):
        """
        Function to collate the training or test data into blocks that have a
        size corresponding to the number of used input frames

        Returns
        -------
        None.

        """
        self.collated_data = []
        self.collated_labels = []
        
        if not self.use_clusters:
            if self.nframes == 1:
                self.collated_data = np.expand_dims(self.data, axis=1)
                self.collated_labels = torch.from_numpy(self.labels)
                self.collated_data,\
                self.collated_labels = shuffle(self.collated_data,
                                               self.collated_labels)
                return
                
            
            for i in range(self.nclasses):
                indices = np.argwhere(self.labels == i)
                data_i = list(np.squeeze(self.data[indices]))

                for j in range(len(indices)):
                    collection = random.choices(data_i, k=self.nframes)
                    self.collated_data.append(np.array(collection))
                    self.collated_labels.append(i)

            self.collated_data = np.array(self.collated_data)
            self.collated_labels = np.array(self.collated_labels)
            self.collated_data,\
                self.collated_labels = shuffle(self.collated_data,
                                               self.collated_labels)
            return
        else:
            self.cluster_data()
            return


    def cluster_data(self):
        print('test')
        return
 
    
    def refresh(self):
        print('Refreshing dataset...')
        self.collate_data()
