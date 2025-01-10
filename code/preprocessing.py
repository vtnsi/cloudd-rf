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

output_path = '/opt/ml/processing/output'
#output_path = '/root/ClouddRF_Final/cloudd-rf/data2'

# Spectrum Parameters
sig_types = [['2-ASK',  ['ask', 2], 0],
             ['4-ASK',  ['ask', 4], 1],
             ['8-ASK',  ['ask', 8], 2],
             ['BPSK',   ['psk', 2], 3], 
             ['QPSK',   ['psk', 4], 4],
             ['16-QAM', ['qam', 16], 5],
             ['Tone', ['constant'], 6],
             ['P-FMCW', ['p_fmcw'], 7]]

# Starting sample min and max of any created signal with possible range [0, obs_int].
start_bounds = [0, 0]
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

def float_list(arg):
    return list(map(float, arg.split(',')))

def int_list(arg):
    return list(map(int, arg.split(',')))

def bool_arg(arg):
    return arg == 'True'

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Number of samples per file.
    parser.add_argument("--chunk-size", type=int, default=10)
    # Number of different radio frequency spectrum examples to be created for the dataset.
    parser.add_argument("--sample-size", type=int, default=1000)
    # Percentage of samples that will be used for the training dataset
    parser.add_argument("--train-split", type=float, default=0.7)
    # Percentage of samples that will be used for the validation dataset
    parser.add_argument("--val-split", type=float, default=0.1)
    # Percentage of samples that will be used for the test dataset
    parser.add_argument("--test-split", type=float, default=0.2)
    # The maximum number of signals that will be created in the spectrum (note: if allow_collision=False, the generator will attempt to fit this many signals without overlap in the spectrum until max_trials is reached).
    parser.add_argument("--max-sigs", type=int, default=1)
    # How many tries the generator will attempt to fit the maximum number of signals in the spectrum (note: if allow_collision=True, this parameter doesn't do anything).
    parser.add_argument("--max-trials", type=int, default=100000)
    # Bandwidth min and max of any created signal with possible range (0.0, 1.0).
    # True: Signals can be overlapped in time and/or frequency. False: No overlap in signals but may not generate max_sigs.
    parser.add_argument("--allow-collisions", type=bool_arg, default=False)
    # Image Parameters
    # Image width (in pixels).
    parser.add_argument("--image-width", type=int, default=1000)
    # Image height (in pixels).
    parser.add_argument("--image-height", type=int, default=500)
    # FFT size used to generate the spectrogram image.
    parser.add_argument("--fft-size", type=int, default=256)
    # FFT overlap used to generate the spectrogram image.
    parser.add_argument("--overlap", type=int, default=255)
    # Seed for the random number generator for repeatability (note: script must use all of the same generation parameter bounds and values).
    parser.add_argument("--rand-seed", type=int, default=1337)
    # Number of discrete sensors to generate data for
    parser.add_argument("--num-sensors", type=int, default=4)

    return parser.parse_known_args()

def create_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    
def gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, channel, num_sensors):
    dataset = [[None for x in range(num_samples)] for x in range(num_sensors)]
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
        
        for kk in range(num_sensors):
            dataset[kk][k] = data[kk]

        labels.append(sig_type_num)
    
    return dataset, labels

def chunk_data(chunk_size, num_samples, iq_output_path, label_output_path, dataset, labels):
    CHUNK_SIZE = chunk_size
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
        iq_data = [iq_data.astype(np.csingle)]

        if (label_array is not None):
            label_array = label_array + label
        else:
            label_array = label
            
        if (iq_array is not None):
            iq_array = np.concatenate((iq_array,iq_data),axis=0)
        else:
            iq_array = np.array(iq_data)

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

if __name__ == "__main__":
    
    args, _ = parse_args()
    
    # Initalize Generators
    rng = np.random.default_rng(args.rand_seed)

    iq_gen = iqdata_gen(obs_int=np.max(obs_ints), num_sensors=args.num_sensors)
    
    channel_sizes = {
        'train': int(args.sample_size * args.train_split),
        'validation': int(args.sample_size * args.val_split),
        'test': int(args.sample_size * args.test_split)
    }
    
    for channel in ['train','validation','test']:

        print(f'Generating data for {channel} channel')
        
        num_samples = channel_sizes[channel]

        # Create Dataset
        if channel == 'test':
            # Generate Test Data for Each Sig Type
            print(f'Generating data for SIGNAL TYPES')
            for sig_type in sig_types:
                print(f'Generating data for {sig_type[0]}')
                dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_type, rng, iq_gen, channel, args.num_sensors)
                for i in range(1, args.num_sensors+1):
                    print(f'Generating data for sensor {i}')
                    iq_output_path = f"{output_path}/{channel}/{i}/sig_types/{sig_type[0]}/iqdata"
                    label_output_path = f"{output_path}/{channel}/{i}/sig_types/{sig_type[0]}/labeldata"
                    create_path(iq_output_path)
                    create_path(label_output_path)
                    chunk_data(args.chunk_size, num_samples, iq_output_path, label_output_path, dataset[i-1], labels)
                

            # Generate Test Data for Each SNR
            print(f'Generating data for SNRs')
            snrs = range(0,15)
            for snr in snrs:
                print(f'Generating data for {snr}')
                snr_bounds = [(snr, snr),
                (snr, snr),
                (snr, snr),
                (snr, snr)]
                dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, "snr", args.num_sensors)
                for i in range(1, args.num_sensors+1):
                    print(f'Generating data for sensor {i}')
                    iq_output_path = f"{output_path}/{channel}/{i}/snr/{snr}/iqdata"
                    label_output_path = f"{output_path}/{channel}/{i}/snr/{snr}/labeldata"
                    create_path(iq_output_path)
                    create_path(label_output_path)
                    chunk_data(args.chunk_size, num_samples, iq_output_path, label_output_path, dataset[i-1], labels)
                

            # Generate Test Data for Each Center Frequency
            cent_freqs = np.linspace(0.1,0.5,21)
            print(f'Generating data for CENTER FREQUENCIES')
            for cent_freq in cent_freqs:
                print(f'Generating data for {cent_freq}')
                cent_freq_bounds = [(cent_freq, cent_freq),
                        (cent_freq, cent_freq),
                        (cent_freq, cent_freq),
                        (cent_freq, cent_freq)]
                dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, "cent_freqs", args.num_sensors)
                for i in range(1, args.num_sensors+1):
                    print(f'Generating data for sensor {i}')
                    iq_output_path = f"{output_path}/{channel}/{i}/cent_freqs/{cent_freq}/iqdata"
                    label_output_path = f"{output_path}/{channel}/{i}/cent_freqs/{cent_freq}/labeldata"
                    create_path(iq_output_path)
                    create_path(label_output_path)
                    chunk_data(args.chunk_size, num_samples, iq_output_path, label_output_path, dataset[i-1], labels)
        else:
            dataset, labels = gen_data(num_samples, bandwidth_bounds, cent_freq_bounds, snr_bounds, sig_types, rng, iq_gen, channel, args.num_sensors)
            for i in range(1, args.num_sensors+1):
                print(f'Generating {num_samples} samples of data for sensor {i}')
                iq_output_path = f"{output_path}/{channel}/{i}/iqdata"
                label_output_path = f"{output_path}/{channel}/{i}/labeldata"
                create_path(iq_output_path)
                create_path(label_output_path)
                chunk_data(args.chunk_size, num_samples, iq_output_path, label_output_path, dataset[i-1], labels)

    print("Finished running processing job")
