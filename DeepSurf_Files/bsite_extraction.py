#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:00:46 2020

@author: smylonas
"""

import numpy as np
from sklearn.cluster import MeanShift

class Bsite_extractor():
    """
    Steps from 9 to 15 of DeepSurf's algorithm. After getting the ligandability
    scores from he 3D CNN this class is the responsible for returning actual biding 
    sites by clustering the points and ranking them.
    """
    def __init__(self, lig_thres, bw=15):
        # Initialize the binding site extractor
        # lig_thres: threshold for ligandability score
        # bw: bandwidth for the MeanShift algorithm
        self.T = lig_thres
        self.ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=False, n_jobs=4)
    
    def _cluster_points(self, prot, lig_scores):
        # Filter and cluster surface points of the protein based on ligandability scores
        T_new = self.T
        # Adjust the threshold to ensure at least 10 points have scores greater than or equal to the threshold
        while sum(lig_scores >= T_new) < 10 and T_new > 0.3001:
            T_new -= 0.1 

        # Filter surface points and their scores based on the adjusted threshold
        filtered_points = prot.surf_points[lig_scores > T_new]
        filtered_scores = lig_scores[lig_scores > T_new]
        
        # If there are fewer than 5 points, a significant binding site cannot be formed
        if len(filtered_points) < 5:
            return () 

        # Apply the MeanShift clustering algorithm to the filtered points
        clustering = self.ms.fit(filtered_points)
        labels = clustering.labels_
        
        # Get unique clusters and their frequencies
        unique_l, freq = np.unique(labels, return_counts=True)
    
        # Keep only clusters with at least 5 points
        if len(unique_l[freq >= 5]) != 0:
            unique_l = unique_l[freq >= 5]
        else:
            return ()
        
        # Discard the "unclustered" cluster (labeled as -1)
        if unique_l[0] == -1:
            unique_l = unique_l[1:]    
        
        # Group the points and their corresponding scores into clusters
        clusters = [(filtered_points[labels == l], filtered_scores[labels == l]) for l in unique_l]
        
        return clusters
        
    def extract_bsites(self, prot, lig_scores):
        # Extract binding sites from the protein based on ligandability scores
        clusters = self._cluster_points(prot, lig_scores)
        
        # If no clusters were found, inform and exit the function
        if len(clusters) == 0:
            print('No binding site found')
            return
        
        # Add the clusters as binding sites to the protein
        for cluster in clusters:
            prot.add_bsite(cluster)
        
        # Sort and save the binding sites in the protein
        prot.sort_bsites()
        prot.write_bsites()
