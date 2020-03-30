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


def load_data(filename, kfold=3, seed=333, split='random'):
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
    valid_idx = data['hasValidLabel'].flatten() == 1
    balanced_idx = data['isBalanced'].flatten() == 1
    # indices now gives a subset of the data set that contains only valid
    # pressure frames and the same number of frames for each class
    indices = np.logical_and(valid_idx, balanced_idx)
    pressure = np.transpose(data['pressure'], axes=(0, 2, 1))
    # Prepare the data the same way as in the paper
    pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
    # Reshape the data to use sklearn utility functions
    pressure = np.reshape(pressure, (-1, 32*32))
    object_id = data['objectId'].flatten()
    
    if split == 'original':
        split_idx = data['splitId'].flatten() == 0
        train_indices = np.logical_and(indices, split_idx)
        pressure_train = pressure[train_indices]
        
        train_data = pressure_train
        train_labels = object_id[train_indices]

        split_idx = data['splitId'].flatten() == 1
        test_indices = np.logical_and(indices, split_idx)
        pressure_test = pressure[test_indices]
        
        test_data = pressure_test
        test_labels = object_id[test_indices]
        
        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
        
        #Just to test if the accuracy in the test set itself stays high
        # train_data, test_data,\
        #     train_labels, test_labels = train_test_split(pressure_train,
        #                                                  train_labels,
        #                                                  test_size=0.2,
        #                                                  random_state=seed,
        #                                                  shuffle=True,
        #                                                  stratify=train_labels)
        #_____________________________________________________________________#

        return train_data, train_labels, test_data, test_labels

    elif split == 'random':
        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]

        pressure = pressure[indices]
        object_id = object_id[indices]
    
        train_data, test_data,\
            train_labels, test_labels = train_test_split(pressure, object_id,
                                                         test_size=0.306,
                                                         random_state=seed,
                                                         shuffle=True,
                                                         stratify=object_id)
        #print(train_data.shape, train_labels.shape)
        skf = StratifiedKFold(n_splits=kfold, shuffle=True,
                              random_state=seed+1)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        skf_gen = skf.split(train_data, train_labels)
        
        # # Add the rest of the valid data to the test set
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
    
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
        self.augment = augment


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        pressure = torch.from_numpy(self.collated_data[idx])
        if self.augment:
            noise = torch.randn_like(pressure) * 0.015
            pressure += noise
        object_id = torch.LongTensor([int(self.collated_labels[idx])])
        return self.dummyrow, self.dummyimage, pressure, object_id


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
                self.collated_labels = self.labels
                # shuffling is taken care of by the torch.utils.data.DataLoader
                self.collated_data,\
                self.collated_labels = shuffle(self.collated_data,
                                                self.collated_labels)
                return
                
            
            for i in range(self.nclasses):
                # Get all pressure frames from the same class
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
        """
        Function to group the data into clusters in order to maximize the
        information content of an input to the network.
        Not yet implemented because it doesn't seem to be necessary

        Returns
        -------
        None.

        """
        print('test')
        return
 
    
    def refresh(self):
        print('Refreshing dataset...')
        self.collate_data()
