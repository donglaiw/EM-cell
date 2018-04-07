import numpy as np
from scipy.ndimage import gaussian_filter, label, distance_transform_edt

def seg_DoG(img, filter1_size = [6,6,0.8], filter2_size = [15,15,2],
            dog_thres=8, dust_thres=3000, resolution=(4,4,30),
            dt1_thres=300, dt2_thres=300):

    # Computing DOG
    dog = gaussian_filter(-img.astype(np.float32), filter1_size) - \
          gaussian_filter(-img.astype(np.float32), filter2_size)

    # Labeling to remove small blobs
    labels, count = label(dog > dog_thres)
    counts = np.bincount(labels.flatten())
    counts[0] = 0
    mask = counts[labels] < dust_thres
    # computing first distance transform 
    dt = distance_transform_edt(mask, sampling=resolution)
    # computing second distance transform 
    reverse_dt = distance_transform_edt(dt < dt1_thres, sampling=resolution)
    seg, count = label(reverse_dt < dt2_thres)
    return seg
