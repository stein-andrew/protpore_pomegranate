from pomegranate.distributions import Normal, Uniform
from pomegranate.hmm import DenseHMM
import numpy as np
import pandas as pd
import os
import sys
import time
import h5py
from scipy import stats

'''
version 0.0.1
Author: Andrew Stein and Neda Ghohabi
Date: 2024-08-28

This is the updated design of the /protpore repository, which will be built on the pomegranate HMM module as opposed 
to the deprecated YAHMM module. This will be a complete rewrite of the previous code, and will be designed to be more
directed towards peptide analysis as opposed to nucleic acid HMM analysis.

TODO:
- Read in the data from .txt as the following format: Columns: ['Index', 'Gaussian Mean', 'Gaussian Std Dev']
- Create a HMM model with the following states: ['Match', 'Skip', 'Backslip']
- Train the model on the data
- Test the model on the data
- Implement a function to find the most likely path through the model
- Implement a function to find the most likely path through the model given a sequence

'''

# Wrapper function to time the execution of the HMM
def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start}")
        return result
    return wrapper

class OuterPeptideHMM:
    """
    Designed to be the outer HMM, which will be used to segment the signal into different regions, each of which
    is then passed to the segmenter for further analysis. The states of this HMM are 'Match' and 'YY'.  Here, each match
    state represents a region of the signal that is of interest, while the 'YY' state represents a tyrosine dip in the
    signal. The transitions between states are unknown, and will be learned from the raw data. Outputs for this HMM are
    regions of ionic current that are of interest, and the tyrosine dip in the signal.
    """

    def __init__(self):
        self.model = None
        self.data = None
        self.states = ['Match', 'YY']
        self.transitions = None
        self.emissions = None
        self.initial_probs = None

    # Function to load the data from a file
    def read_fast5_range(self, file_path, channel, start_index, stop_index):
        """
        Yields data in chunks from a specified channel between given start and stop indices.

        :param file_path: Path to the FAST5 file.
        :param channel: The channel name from which to read the data.
        :param start_index: The starting index of the data range to read.
        :param stop_index: The stopping index of the data range to read.
        :yield: The chunk of adjusted signal data and metadata from the specified channel.
        """
        with h5py.File(file_path, 'r') as f:
            # Ensure the specified channel exists
            if channel not in f['Raw']:
                raise ValueError(f"Channel '{channel}' not found in the file.")

            # Validate indices
            total_length = f['Raw'][channel]['Signal'].shape[0]
            if start_index < 0 or stop_index > total_length:
                raise ValueError(f"Start ({start_index}) or stop ({stop_index}) indices are out of bounds.")
            if start_index >= stop_index:
                raise ValueError("Start index must be less than stop index.")

            # Collect metadata
            meta_attrs = f['Raw'][channel]['Meta'].attrs
            meta = {
                'description': meta_attrs['description'],
                'digitisation': meta_attrs['digitisation'],
                'offset': meta_attrs['offset'],
                'range': meta_attrs['range'],
                'sample_rate': meta_attrs['sample_rate']
            }

            # Extract the raw signal in the specified range
            signal = f['Raw'][channel]['Signal'][start_index:stop_index]

            # Convert to a NumPy array with a smaller data type if possible
            signal = np.array(signal, dtype=np.float32)

            # Adjust the raw signal in place
            digitisation = meta['digitisation']
            offset = meta['offset']
            range_ = meta['range']
            signal += offset
            signal *= (range_ / digitisation)

            # Yield the chunk of adjusted signal data and metadata
            yield signal, meta

    # Function to create the HMM model
    def create_model(self):
        """
        Creates the HMM model with the specified states and transitions.
        """
        # Create the model
        model = DenseHMM([Normal(), Uniform()],
                         edges=[(0, 1), (1, 0)])
        self.model = model

        return model
class InnerPeptideHMM:
    def __init__(self):
        self.model = None
        self.data = None
        self.states = ['Match', 'Skip', 'Backslip']
        self.transitions = None
        self.emissions = None
        self.initial_probs = None

    # Function to load the data from a file
    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep="\t")
        data.columns = ['Index', 'Mean', 'Std Dev']
        return data

    # Function to create the HMM model



