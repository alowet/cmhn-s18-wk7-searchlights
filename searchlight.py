# Run a whole brain searchlight

# Import libraries
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from scipy.spatial.distance import euclidean

# Import additional libraries you need

# What subject are you running
sub = 'sub-01'
output_name = ('%s_whole_brain_SL.nii.gz' % (sub))

# Get information
nii = nib.load(('%s_input.nii.gz' % (sub)))
affine_mat = nii.affine  # What is the data transformation used here
dimsize = nii.header.get_zooms()

# Preset the variables
data = nii.get_data()
mask = nib.load(('%s_mask.nii.gz' % (sub))).get_data()
bcvar = np.load(('%s_labels.npy' % (sub)))
sl_rad = 1
max_blk_edge = 5
pool_size = 1

# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute([data], mask)

# Broadcast variables
sl.broadcast(bcvar)

# Set up the kernel, in this case an SVM
def calc_svm(data, mask, myrad, bcvar):
    
    # Pull out the data
    data4D = data[0]
    labels = bcvar
    
    bolddata_sl = data4D.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], data[0].shape[3]).T

    # Check if the number of voxels is what you expect.
    # print("Input data reshaped: " + str(bolddata_sl.shape))
    # print("Input mask:\n" + str(mask) + "\n")
    
    # t1 = time.time()
    clf = SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, bolddata_sl, labels, cv=3)
    accuracy = scores.mean()
    # t2 = time.time()
    
    # print('Kernel duration: ' + str(t2 - t1))
    
    return accuracy

# Run the searchlight analysis
sl_result = sl.run_searchlight(calc_svm, pool_size=pool_size)

# print("End SearchLight")

# end_time = time.time()

# Print outputs
# print("Accuracy: " + str(sl_result[mask==1]))
# print('Searchlight duration: ' + str(end_time - begin_time))

# Only save the data if this is the first core
if rank == 0: 

    # Convert the output into what can be used
    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0  # If there are nans we want this

    # Save the volume
    sl_nii = nib.Nifti1Image(sl_result, affine_mat)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save
    
    print('Finished searchlight')