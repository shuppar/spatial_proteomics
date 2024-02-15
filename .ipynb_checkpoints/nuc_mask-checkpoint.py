#!/usr/bin/env python3
import numpy as np
import cv2
from skimage import morphology, segmentation, measure, color
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

# A few things to note (Important)
# You will have to take a blank fluorescein image to correct for uneven illumination.
# So you might need to change the disk size in strel in the second section.
# Calculate the minimum and maximum nuclear sizes and make changes
# accordingly in the section devoted to it.

# setting some parameters (these need to be adapted ad libitum)
d_size = 10 # radius of disk in pixels for preprocessing.
d_size1 = 18 # radius of disk for smoothening pixels.
max_area = 35000
min_area = 8000
max_roundness = 1.25
min_roundness = 0.45


# Shuppar Script for Intensity plots.

# B = cv2.imread('xxx.tif', cv2.IMREAD_GRAYSCALE) # Protein 1
# C = cv2.imread('yyy.tif', cv2.IMREAD_GRAYSCALE) # protein 2
A = cv2.imread('DAPI1.tif', cv2.IMREAD_GRAYSCALE)
DAPI = cv2.imread('DAPI1.tif', cv2.IMREAD_GRAYSCALE)

# Nonuniform illumination correction. Function saved in home/MatlabCode. Copy of Image Analyst's code. (August 2, 2016)
# A = BackgroundCorrect(A, BlankImage.tif);         # Take a blank fluorescein image.
# B = BackgroundCorrect(B, BlankImage.tif);
# C = BackgroundCorrect(C, BlankImage.tif);

A = cv2.equalizeHist(A)  # Some local adjustments, to be able to detect the dimmer cells.
A = cv2.morphologyEx(A, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_size, d_size)))
A = cv2.clearBorder(A)  # Eliminating the objects on the borders.
# Removing pixels smaller than the given size.
A = cv2.wienerFilter(A, (d_size, d_size))

# Removing the problem of oversegmentation.
se = morphology.disk(d_size1)
Ao = morphology.opening(A, se)

Ae = morphology.erosion(A, se)
Aobr = morphology.reconstruction(Ae, A)
Aoc = morphology.closing(Ao, se)
Aobrd = morphology.dilation(Aobr, se)
Aobrcbr = morphology.reconstruction(Aobrd, Aobr)

A = Aobrcbr

# From here it continues as before.
bw = A > threshold_otsu(A)  # graythresh was giving black image. Hence put a random value, works nevertheless for this case.

bw1 = morphology.remove_small_holes(bw)
bw2 = morphology.opening(bw1, morphology.disk(18))  # Morphological opening.
bw3 = morphology.remove_small_objects(bw2, min_size=100)  # Removing cells with less than 100 pixels.

bw3_perim = segmentation.find_boundaries(bw3)
overlay = color.label2rgb(bw3_perim, image=A, colors=[(1, 0.3, 0.3)])

# Discovering Putative nucleus centroid.
maxs = ndimage.maximum_filter(A, size=5)
maxs = morphology.closing(maxs, se)
maxs = morphology.remove_small_holes(maxs)
maxs = morphology.remove_small_objects(maxs, min_size=100)
overlay1 = color.label2rgb(bw3_perim | maxs, image=A, colors=[(1, 0.3, 0.3)])

# Modifying the image so that the background pixels and the extended maxima pixels are forced to be the only minima in the image.
Jc = 255 - A
A_mod = ndimage.morphological_reconstruction(Jc, (bw3 | maxs).astype(int))

# Watershed algorithm
distance = ndimage.distance_transform_edt(A_mod)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=A)
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=A_mod)

# Counting cells and removing under- and over-segmented nuclei.
regions = measure.regionprops(labels, intensity_image=DAPI)

for props in regions:
    if props.area <= max_area and props.area >= min_area and \
            (props.perimeter) ** 2 / (4 * np.pi * props.area) <= max_roundness and \
            (props.perimeter) ** 2 / (4 * np.pi * props.area) >= min_roundness and \
            (4 * props.area) / (np.pi * (props.major_axis_length) ** 2) <= max_roundness and \
            (4 * props.area) / (np.pi * (props.major_axis_length) ** 2) >= min_roundness:
        cm1 = props.label
    else:
        cm2 = props.label
        labels[props.coords[:, 0], props.coords[:, 1]] = 0

# Cell counting:
num_cells = np.max(labels)

# Overlaying the cells detected over the original image.
mask = labels > 0
overlay2 = color.label2rgb(labels, image=A, bg_label=0, colors=[(.5, .8, .3)])

# Writing Data
# Assuming intensity.dat is a file to write data
with open('Intensity.dat', 'a') as f:
    for props in regions:
        idx1 = props.mean_intensity
        idx2 = props.area
        # idx3 = props.mean_intensity
        # idx6 = props.mean_intensity
        # idx4 = np.mean(ndimage.standard_deviation(C[props.coords[:, 0], props.coords[:, 1]])) * props.area
        # idx5 = np.mean(ndimage.entropy(C[props.coords[:, 0], props.coords[:, 1]])) * props.area
        # idx7 = np.mean(ndimage.standard_deviation(B[props.coords[:, 0], props.coords[:, 1]])) * props.area
        # idx8 = np.mean(ndimage.entropy(B[props.coords[:, 0], props.coords[:, 1]])) * props.area
        
        DI = idx1 * idx2  # DAPI intensity
        # HI = idx3 * props.area  # H2AX intensity
        # KI = idx6 * props.area  # Ki67 Intensity
        
        f.write(f"{DI}\n") #writing DNA intensity/content into a file
# DAPI_total, Ki67_total,H2A_Total
    
