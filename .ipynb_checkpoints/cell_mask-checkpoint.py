import numpy as np
import cv2
from scipy.ndimage import label, distance_transform_edt
from skimage import exposure, morphology, filters, segmentation, measure


def CMask(CellImage, NucleusImage, InterestNuc, **kwargs):
    # Initialization
    Dobrcbr = CellImage.copy()
    DAPI = NucleusImage.copy()
    In = InterestNuc.copy()

    # Anisotropic diffusion
    DAPI = exposure.rescale_intensity(DAPI, out_range=(0, 255)).astype(np.uint8)
    DAPI = cv2.dilate(DAPI, None, iterations=1)  # Dilate for better diffusion
    DAPI = cv2.normalize(DAPI, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    Dobrcbr = cv2.GaussianBlur(Dobrcbr, (0, 0), 0.35)
    Dobrcbr = exposure.rescale_intensity(Dobrcbr, out_range=(0, 255)).astype(np.uint8)

    # Getting nuclear mask
    se = morphology.disk(4)
    Ae = morphology.binary_erosion(DAPI, se)
    Aobr = morphology.reconstruction(Ae, DAPI)
    Aobrd = morphology.binary_dilation(Aobr, se)
    Aobrcbr = morphology.reconstruction(~Aobrd, ~Aobr)
    Aobrcbr = ~Aobrcbr

    # Watershed for final segmentation
    distance = -distance_transform_edt(~Aobrcbr)
    markers = segmentation.extended_local_minima(distance, indices=False, footprint=np.ones((3, 3, 3)))
    markers_labelled = label(markers)
    markers_labelled[In == 1] = 1  # Mark the nucleus of interest
    L = segmentation.watershed(distance, markers_labelled, mask=~Aobrcbr)

    return L


# Example usage:
# CellMask = CMask(cropPhal, nucmark, nucInt, thresh='local', s1='Mean', s2='Mean', meth='m2', diskR=11, g1=4, g2=1.5)
