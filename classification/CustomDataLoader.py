#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:37:27 2020

@author: fabian
"""


import numpy as np
import random
from sklearn.utils import shuffle
import torch.utils.data as data
import torch        
from imblearn import over_sampling


class CustomDataLoader(data.Dataset):
    def __init__(self, data, labels, augment=False, nframes=5,
                  use_clusters=False, nclasses=27, oversample=False):
        self.nframes = nframes
        self.nclasses = nclasses
        self.data = data
        self.labels = labels
        self.use_clusters = use_clusters
        self.dummyrow = torch.zeros((32,1))
        self.dummyimage = torch.zeros((3, 1, 1))
        self.augment = augment
        self.oversample = oversample
        if oversample:
            self.original_data = data.reshape(len(labels), 32*32)
            self.original_labels = labels
            self.balance_data()
        self.collate_data()

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        pressure = torch.from_numpy(self.collated_data[idx])
        if self.augment:
            noise = torch.randn_like(pressure) * 0.015#0.015
            pressure += noise
        object_id = torch.LongTensor([int(self.collated_labels[idx])])
        return self.dummyrow, self.dummyimage, pressure, object_id


    def collate_data(self):
        """
        Function to collate the training or test data into blocks that are
        sized corresponding to the number of used input frames
        e.g. if 4 input frames are used, one block has the shape (4,32,32)

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
        
    def balance_data(self):
        # Randomize for every refresh call
        seed = random.randint(0,1000)
        neighbors = random.randint(2,8)
        clusters = random.randint(14,20)
        # step = random.random()
        oversampler = over_sampling.KMeansSMOTE(random_state=seed,
                                                kmeans_estimator=clusters,
                                                k_neighbors=neighbors)
        # oversampler = over_sampling.SVMSMOTE(random_state=seed, out_step=step,
        #                                      k_neighbors=5, m_neighbors=10)
        resampled_data,\
            resampled_labels = oversampler.fit_resample(self.original_data,
                                                        self.original_labels)
        self.data = resampled_data.reshape((len(resampled_labels), 32, 32))
        self.labels = resampled_labels


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
        if self.oversample:
            self.balance_data()
        self.collate_data()
