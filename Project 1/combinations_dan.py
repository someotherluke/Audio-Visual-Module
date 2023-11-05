# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 00:04:26 2023

@author: haw17cju
"""
from Feature_Extraction import extract_features
import itertools

#Define the possible variables
possible_filter_types = ["Mel", "Bark"]
possible_filter_shapes = ["Square", "Triangle"]
possible_overlaps = [1/2, 1/3]          #1/4 not included for time
possible_frame_sizes = [512, 1024]      #2048 not included for time
possible_energy_included = [1]          #1 for dan, 0 for luke
possible_temporal_included = [1]        #1 for dan, 0 for luke
possible_no_bins = [32,64,128]
possible_truncate_no_bin = [16,24]      #8 not included for time

#Generate a matrix of all possible combinations
combinations = list(
    itertools.product(
        possible_filter_types,
        possible_filter_shapes,
        possible_overlaps,
        possible_frame_sizes,
        possible_energy_included,
        possible_temporal_included,
        possible_no_bins,
        possible_truncate_no_bin
    )
)

#Iterate through the combinations
def combination():
    for combination in combinations:
        try:
            filter_type = combination[0]
            filter_shape = combination[1]
            overlap = combination[2]
            frame_sizes = combination[3]
            energy_included = combination[4]
            temporal_included = combination[5]
            no_bins = combination[6]
            truncate_no_bins = combination[7]
            
            extract_features(filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)
            print(filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)
        except:
            print("error occured with settings", filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)

if __name__ == '__main__':
    combination()