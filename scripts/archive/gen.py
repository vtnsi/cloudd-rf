import argparse
from pathlib import Path

import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

# Datagen Imports
from cloudd_rf.datagen.iqdata_gen import iqdata_gen
from cloudd_rf.datagen.burst_def import burst_def


bucket = 'summer-team-bucket'
s3_output1_path = f's3://{bucket}/team1'
s3_output2_path = f's3://{bucket}/team2'
s3_output3_path = f's3://{bucket}/team3'
s3_output4_path = f's3://{bucket}/team4'
output_path_1 = s3_output1_path
output_path_2 = s3_output2_path
output_path_3 = s3_output3_path
output_path_4 = s3_output4_path

output_paths = ['/root/ClouddRF_Final/cloudd-rf/data']

# Spectrum Parameters
rand_seed = 1337                                            # Seed for the random number generator for repeatability (note: script must use all of the same generation parameter bounds and values).
max_sigs = 1                                                # The maximum number of signals that will be created in the spectrum (note: if allow_collision=False, the generator will attempt to fit this many signals without overlap in the spectrum until max_trials is reached).
max_trials = 100000                                          # How many tries the generator will attempt to fit the maximum number of signals in the spectrum (note: if allow_collision=True, this parameter doesn't do anything).
start_bounds = [0, 0]                                       # Starting sample min and max of any created signal with possible range [0, obs_int].
sig_types = [['2-ASK',  ['ask', 2], 0],
             ['4-ASK',  ['ask', 4], 1],
             ['8-ASK',  ['ask', 8], 2],
             ['BPSK',   ['psk', 2], 3], 
             ['QPSK',   ['psk', 4], 4],
             ['16-QAM', ['qam', 16], 5],
             ['Tone', ['constant'], 6],
             ['P-FMCW', ['p_fmcw'], 7]]
obs_ints = [2048, 
            1024, 
            512, 
            256]
bandwidth_bounds = [(0.1, 0.5),
                    (0.25, 0.5),
                    (0.25, 0.5),
                    (0.25, 0.5)]
cent_freq_bounds = [(-0.01, 0.01),
                    (-0.01, 0.01),
                    (-0.05, 0.05),
                    (-0.01, 0.01)]
snr_bounds = [(5, 15),
              (0, 20),
              (5, 20),
              (5, 20)]
allow_collisions = False                                       # True: Signals can be overlapped in time and/or frequency. False: No overlap in signals but may not generate max_sigs.

# Image Parameters
image_width = 1000                                          # Image width (in pixels).
image_height = 500                                          # Image height (in pixels).
fft_size = 256                                              # FFT size used to generate the spectrogram image.
overlap = 255                                               # FFT overlap used to generate the spectrogram image.
num_sensors = 1

# Initalize Generators
rng = np.random.default_rng(rand_seed)

iq_gen = iqdata_gen(obs_int=np.max(obs_ints), num_sensors=num_sensors)

sample_size= 150000
train_split = 0.7
val_split=0.2
test_split=0.1

channel_sizes = {
    'train': int(sample_size * train_split),
    'validation': int(sample_size * val_split),
    'test': int(sample_size * test_split)
}

def create_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, channel):
    dataset1 = []
    dataset2 = []
    dataset3 = []
    dataset4 = []
    labels = []
    for k in range(num_samples):
        if channel == "test":
            sig_type_num = sig_types[2]
        else:
            sig_type_num = rng.choice(len(sig_types))

        burst_list = []
        for kk in range(num_sensors):
            burst_list.append(burst_def(rng.uniform(bandwidth_bounds[kk][0], bandwidth_bounds[kk][1]), rng.uniform(cent_freq_bounds[kk][0], cent_freq_bounds[kk][1]), obs_ints[kk], rng.uniform(snr_bounds[kk][0], snr_bounds[kk][1])))

        if channel == "test":
            data, burst_list = iq_gen.gen_iq(sig_types, burst_list)
        else:
            data, burst_list = iq_gen.gen_iq(sig_types[sig_type_num], burst_list)
        dataset1.append(data[0])
        dataset2.append(data[1])
        dataset3.append(data[2])
        dataset4.append(data[3])
        labels.append(sig_type_num)
    dataset = [dataset1, dataset2, dataset3, dataset4]
    return dataset, labels
    
def chunk_data(num_samples, iq_output_path, label_output_path, dataset, labels):
    CHUNK_SIZE = 5000
    iq_array = None
    label_array = None
    CHUNK = 0
    
    start_time = time.time()
    for k in range(num_samples):
        # Create Radio Frequency Signal Example
        # Usage Notes: 
        #   - Generated signal is a vector of complex radio frequency samples (i.e. each sample is of the form A+jB).
        #   - For neural network training and testing, you will need to convert this complex data to a real format.
        #       - Option 1: Real and Imaginary components of each sample are stored as seperate 'channels'.
        #       - Option 2: Real and Imaginary components of each sample are stored as seperate 'rows'.
        label = [labels[k]]
        iq_data = dataset[k]
        iq_data = iq_data.astype(np.csingle)
        
        if (label_array is not None):
            label_array = label_array + label
        else:
            label_array = label

        if (iq_array is not None):
            iq_array = np.concatenate([iq_array,iq_data],axis=0)
        else:
            iq_array = iq_data
        
        if k > 0 and (k+1) % CHUNK_SIZE == 0:   
            CHUNK += 1
            write_chunk(iq_array, label_array, CHUNK, iq_output_path, label_output_path)
            iq_array = None
            label_array = None
            print('Finished chunk ' + str(CHUNK) + ' of ' + str(int(num_samples/CHUNK_SIZE)) + ' generated. Time taken: ' + str(time.time()-start_time))
            start_time = time.time()

def write_chunk(iq_data, labels, chunk, iq_output_path, label_output_path):
    # Save Radio Frequency Data to File 
    # Usage Notes: 
    #   - This is the data that acts as the input for neural network training. 
    #   - File is in the numpy csingle format (https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.csingle) and will need to be loaded from file as such.
    iqdata_file_name = f'{iq_output_path}/example_{str(chunk)}.dat'
    iq_data.tofile(iqdata_file_name)
    
    # Save Radio Frequency Spectrogram Image to File
    # Usage Notes: 
    #   - This is for optional visualization purposes only and not to be used as the input for neural network training. 
    #   - Can be safely commented out to speed up data generation times.
    # imdata_file_name = f'{im_output_path}/example_{str(chunk)}.png'
    # im_gen.gen_image(imdata_file_name, burst_metadata, iq_data, False)
    # pyplot.close()

    # Save Radio Frequency Metadata to File
    # Usage Notes: 
    #   - This is all of the metadata that will be useful for neural network training and testing.
    #   - For training, only the 'Signal Type' field is needed.
    #   - For testing, all fields will be useful for quantifying performance as a function of the signal parameters (e.g. performance as a function of 'SNR', 'Duration', 'Signal Type', etc.).
    label_file_name = f'{label_output_path}/example_{str(chunk)}.csv'
    fid = open(label_file_name, 'w', encoding='UTF8')
    writer = csv.writer(fid)

    header = ['Label']
    writer.writerow(header)
    for label in labels:
        data = [label]
        writer.writerow(data)

for channel in ['train','validation','test']:
#for channel in ['test']:
    print(f'Generating data for {channel} channel')
        
    num_samples = channel_sizes[channel]

    # Create Dataset
    if channel == 'test':
        # Generate Test Data for Each Sig Type
        print(f'Generating data for SIGNAL TYPES')
        for sig_type in sig_types:
            dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_type, rng, iq_gen, channel)
            for i in range(num_sensors):
                iq_output_path = f"{output_paths[i]}/{channel}/sig_types/{sig_type[0]}/iqdata"
                label_output_path = f"{output_paths[i]}/{channel}/sig_types/{sig_type[0]}/labeldata"
                create_path(iq_output_path)
                create_path(label_output_path)
                chunk_data(num_samples, iq_output_path, label_output_path, dataset[i], labels)
            print(f'Generating data for {sig_type[0]}')
        
        
        # Generate Test Data for Each SNR
        snrs = range(0,15)
        print(f'Generating data for SNRs')
        for snr in snrs:
            snr_bounds = [(snr, snr),
              (snr, snr),
              (snr, snr),
              (snr, snr)]
            dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, "snr")
            for i in range(num_sensors):
                iq_output_path = f"{output_paths[i]}/{channel}/snr/{snr}/iqdata"
                label_output_path = f"{output_paths[i]}/{channel}/snr/{snr}/labeldata"
                create_path(iq_output_path)
                create_path(label_output_path)
                chunk_data(num_samples, iq_output_path, label_output_path, dataset[i], labels)
            print(f'Generating data for {snr}')
            
        # Generate Test Data for Each Center Frequency
        cent_freqs = np.linspace(0.1,0.5,21)
        print(f'Generating data for CENTER FREQUENCIES')
        for cent_freq in cent_freqs:
            cent_freq_bounds = [(cent_freq, cent_freq),
                    (cent_freq, cent_freq),
                    (cent_freq, cent_freq),
                    (cent_freq, cent_freq)]
            dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, "cent_freqs")
            for i in range(num_sensors):
                iq_output_path = f"{output_paths[i]}/{channel}/cent_freqs/{cent_freq}/iqdata"
                metadata_output_path = f"{output_paths[i]}/{channel}/cent_freqs/{cent_freq}/labeldata"
                create_path(iq_output_path)
                create_path(metadata_output_path)
                chunk_data(num_samples, iq_output_path, label_output_path, dataset[i], labels)
            print(f'Generating data for {cent_freq}')
    else:
        dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, channel)
        for i in range(num_sensors):
            iq_output_path = f"{output_paths[i]}/{channel}/iqdata"
            label_output_path = f"{output_paths[i]}/{channel}/labeldata"
            create_path(iq_output_path)
            create_path(label_output_path)
            chunk_data(num_samples, iq_output_path, label_output_path, dataset[i], labels)
    

        #print('Example ' + str(k+1) + ' of ' + str(channel_sizes[channel]) + ' generated. Time taken: ' + str(time.time()-start_time))
    
print("Finished running processing job")