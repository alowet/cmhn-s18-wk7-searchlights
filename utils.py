# Import modules
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
import scipy.io
from scipy import stats
from sklearn import preprocessing

# Make a function to load the mask data
def load_data(directory, subject_name, mask_name='', num_runs=3,zscore_data=False):
    
    maskdir = (directory + subject_name + "/preprocessed/masks/")

    # Cycle through the masks
    print ("Processing Start ...")
    
    # If there is a mask supplied then load it now
    if mask_name is not '':
        maskfile = (maskdir + "%s_ventral_%s_locColl_to_epi1.nii.gz" % (subject_name, mask_name))

        mask = nib.load(maskfile)
        print ("Loaded %s mask" % (mask_name))

    # Cycle through the runs
    for run in range(1, num_runs + 1):
        epi_in = (directory + subject_name + "/preprocessed/loc/%s_filtered2_d1_firstExampleFunc_r%d.nii" % (subject_name, run))
        print(epi_in)

        # Load in the fmri data
        epi_data = nib.load(epi_in)
        
        # Mask the data if necessary
        if mask_name is not '':
            nifti_masker = NiftiMasker(mask_img=mask)
            epi_mask_data = nifti_masker.fit_transform(epi_data);
            epi_mask_data = np.transpose(epi_mask_data)
        else:
            # Do a whole brain mask 
            if run == 1:
                mask = compute_epi_mask(epi_data).get_data() # Compute mask from epi
            else:
                mask *= compute_epi_mask(epi_data).get_data() # Get the intersection mask (so that the voxels are the same across runs)  
            
            # Reshape all of the data (not great for memory
            epi_mask_data = epi_data.get_data().reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], epi_data.shape[3])

        # Transpose and Z-score (Standardize) the data  
        if zscore_data == True:
            scaler = preprocessing.StandardScaler().fit(epi_mask_data)
            preprocessed_data =  scaler.transform(epi_mask_data)
        else:
            preprocessed_data = epi_mask_data
        
        # Concatenate the data
        if run == 1:
            concatenated_data = preprocessed_data
        else:
            concatenated_data = np.hstack((concatenated_data, preprocessed_data))
    
    # Now zero out all of the voxels outside of the mask across all runs
    if mask_name is '':
        mask_vector = np.nonzero(mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], ))[0]
        concatenated_data = concatenated_data[mask_vector, :]
        
    # Return the list of mask data
    return concatenated_data, mask

# Make a function for loading in the labels
def load_labels(directory, subject_name):
    stim_label = [];
    stim_label_concatenated = [];
    for run in range(1,4):
        in_file= (directory + subject_name + '/ses-day2/design_matrix/' + "%s_localizer_0%d.mat" % (subject_name, run))

        # Load in data from matlab
        stim_label = scipy.io.loadmat(in_file);
        stim_label = np.array(stim_label['data']);

        # Store the data
        if run == 1:
            stim_label_concatenated = stim_label;
        else:       
            stim_label_concatenated = np.hstack((stim_label_concatenated, stim_label))

    print("Loaded ", subject_name)
    return stim_label_concatenated


# Convert the TR
def label2TR(stim_label, num_runs, TR, TRs_run, events_run):
    # Preset the array with zeros
    stim_label_TR = np.zeros((TRs_run * 3, 1))

    # Cycle through the runs
    for run in range(0, num_runs):

        # Cycle through each element in a run
        for i in range(events_run):

            # What element in the concatenated timing file are we accessing
            time_idx = run * (events_run) + i

            # What is the time stamp
            time = stim_label[2, time_idx]

            # What TR does this timepoint refer to?
            TR_idx = int(time / TR) + (run * (TRs_run - 1))

            # Add the condition label to this timepoint
            stim_label_TR[TR_idx]=stim_label[0, time_idx]
        
    return stim_label_TR

# Create a function to shift the size
def shift_timing(label_TR, TR_shift_size):
    
    # Create a short vector of extra zeros
    zero_shift = np.zeros((TR_shift_size, 1))

    # Zero pad the column from the top.
    label_TR_shifted = np.vstack((zero_shift, label_TR))

    # Don't include the last rows that have been shifted out of the time line.
    label_TR_shifted = label_TR_shifted[0:label_TR.shape[0],0]
    
    return label_TR_shifted


# Extract bold data for non-zero labels.
def reshape_data(label_TR_shifted, masked_data_all):
    label_index = np.nonzero(label_TR_shifted)
    label_index = np.squeeze(label_index)
    
    # Pull out the indexes
    indexed_data = np.transpose(masked_data_all[:,label_index])
    nonzero_labels = label_TR_shifted[label_index] 
    
    return indexed_data, nonzero_labels

# Take in a brain volume and label vector that is the length of the event number and convert it into a list the length of the block number
def blockwise_sampling(eventwise_data, eventwise_labels, events_per_block=10):
    
    # How many events are expected
    expected_blocks = int(eventwise_data.shape[0] / events_per_block)
    
    # Average the BOLD data for each block of trials into blockwise_data

    blockwise_data = np.zeros((expected_blocks, eventwise_data.shape[1]))
    blockwise_labels = np.zeros(expected_blocks)
    
    for i in range(0, expected_blocks):
        start_row = i * events_per_block 
        end_row = start_row + events_per_block - 1 
        
        blockwise_data[i,:] = np.mean(eventwise_data[start_row:end_row,:], axis = 0)
        blockwise_labels[i] = np.mean(eventwise_labels[start_row:end_row])
            
    # Report the new variable sizes
    print('Expected blocks: %d; Resampled blocks: %d' % (expected_blocks, blockwise_data.shape[0]))

    # Return the variables downsampled_data and downsampled_labels
    return blockwise_data, blockwise_labels
