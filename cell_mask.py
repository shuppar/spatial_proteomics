import numpy as np
import cv2
from scipy import ndimage

def CMask(CellImage, NucleusImage, InterestNuc, **kwargs):

    # Initialization
    thresh = kwargs.get('thresh', 'global')
    Rad = kwargs.get('diskR', 7)
    ga1 = kwargs.get('g1', 3)
    ga2 = kwargs.get('g2', 1.5)
    s1 = kwargs.get('s1', 'Mean')
    s2 = kwargs.get('s2', 'Mean')
    method = kwargs.get('meth', 'm1')

    # Works best with the images cropped with the cell in the centre...
    B = CellImage.astype(np.uint16)
    DAPI = NucleusImage.astype(np.uint16)

    DAPI = ndimage.grey_dilation(DAPI, size=(3, 3))  # anisodiff(im, niter, kappa, lambda, option)

    # This affects m2 quite a bit... play around with this if you will.
    B = cv2.GaussianBlur(B, (0, 0), 0.35)
    B = ndimage.grey_dilation(B, size=(3, 3))

    # Getting nuclear mask
    Dobrcbr = B.copy()
    Ae = cv2.erode(DAPI, np.ones((4, 4), np.uint8))
    Aobr = cv2.reconstruct(Ae, DAPI)
    Aobrd = cv2.dilate(Aobr, np.ones((4, 4), np.uint8))
    Aobrcbr = cv2.reconstruct(cv2.bitwise_not(Aobrd), cv2.bitwise_not(Aobr))
    Aobrcbr = cv2.bitwise_not(Aobrcbr)

    conli = cv2.minMaxLoc(Aobrcbr)[0]
    Atemp = cv2.convertScaleAbs(Aobrcbr, alpha=255.0/conli, beta=0.5)
    C = cv2.threshold(Atemp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    Amin = -ndimage.distance_transform_edt(C)

    mask = ndimage.morphology.binary_erosion(Amin < 0, structure=np.ones((3, 3))).astype(np.uint8)
    Amin = ndimage.morphology.binary_dilation(Amin, structure=np.ones((3, 3))).astype(np.uint8)
    somet = ndimage.morphology.binary_dilation(C, structure=np.ones((5, 5))).astype(np.uint8)

    # Processing the cytosol image
    De = cv2.erode(Dobrcbr, np.ones((Rad, Rad), np.uint8))
    Dobr = cv2.reconstruct(De, Dobrcbr)
    Dobrd = cv2.dilate(Dobr, np.ones((Rad, Rad), np.uint8))
    Dobrcbr = cv2.reconstruct(cv2.bitwise_not(Dobrd), cv2.bitwise_not(Dobr))
    Dobrcbr = cv2.bitwise_not(Dobrcbr)

    # Background
    celi = cv2.minMaxLoc(Dobrcbr)[0]
    Cetemp = cv2.convertScaleAbs(Dobrcbr, alpha=255.0/celi, beta=ga1)
    Imp = cv2.threshold(cv2.bitwise_not(Cetemp), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    Imp[somet==1] = 1
    M = cv2.bitwise_not(cv2.connectedComponentsWithStats(cv2.bitwise_not(Imp), connectivity=8)[1] == 0).astype(np.uint8)

    bgm = np.zeros_like(CellImage, dtype=np.uint8)
    bgm1 = cv2.connectedComponentsWithStats(cv2.bitwise_not(M), connectivity=8)[1] != 0
    bgm2 = np.array([np.round(np.mean(np.nonzero(region)[::-1]), 0) for region in cv2.connectedComponentsWithStats(bgm1, connectivity=8)[2]])
    for i, (y, x) in enumerate(bgm2.astype(np.uint64)):
        bgm[x, y] = 1
    bgm1 = cv2.dilate(bgm, np.ones((5, 5), np.uint8))
    bgm2 = ndimage.morphology.binary_erosion(-ndimage.distance_transform_edt(M) < -1, structure=np.ones((3, 3))).astype(np.uint8)

    dist = ndimage.distance_transform_edt(M)
    distl = cv2.connectedComponentsWithStats(cv2.watershed(dist), connectivity=8)[1] == 0
    bgm = distl.astype(np.uint8)
    bgm[bgm1 == 1] = 1
    bgm[bgm2 == 1] = 1

    # Adaptive threshold
    if method == 'm1':
        if thresh == 'global':
            celi = cv2.minMaxLoc(Dobrcbr)[0]
            Cetemp = cv2.convertScaleAbs(Dobrcbr, alpha=255.0/celi, beta=ga1)
            Imp = cv2.threshold(cv2.bitwise_not(Cetemp), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            Imp[somet==1] = 1
            L = cv2.bitwise_not(cv2.connectedComponentsWithStats(cv2.bitwise_not(Imp), connectivity=8)[1] == 0).astype(np.uint8)
        elif thresh == 'local':
            celi = cv2.minMaxLoc(Dobrcbr)[0]
            Dtemp = cv2.convertScaleAbs(Dobrcbr, alpha=255.0/celi, beta=ga1)
            threshp = cv2.adaptiveThreshold(Dtemp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0.7)
            L = cv2.bitwise_not(cv2.connectedComponentsWithStats(cv2.bitwise_not(threshp), connectivity=8)[1] == 0).astype(np.uint8)
            L[somet==1] = 1
        else:
            print('\n\nThreshold should be either "global" or "local"\n\n')

        conli = cv2.minMaxLoc(Dobrcbr)[0]
        Dtemp = cv2.convertScaleAbs(Dobrcbr, alpha=255.0/conli, beta=ga2)
        thresh2 = cv2.adaptiveThreshold(Dtemp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0.7)
        E = cv2.bitwise_not(cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh2), connectivity=8)[1] == 0).astype(np.uint8)
        E[somet==1] = 1

        Dmin = -ndimage.distance_transform_edt(cv2.bitwise_not(E))
        mask1 = ndimage.morphology.binary_erosion(Dmin < 0, structure=np.ones((11, 11))).astype(np.uint8)

        Dmin1 = cv2.bitwise_not(cv2.bitwise_or(Dmin, cv2.bitwise_or(C, bgm)))
        F = cv2.connectedComponentsWithStats(cv2.watershed(Dmin1), connectivity=8)[1]
        L[F == 0] = 0

    elif method == 'm2':
        hy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        hx = hy.T
        Iy = cv2.filter2D(B, -1, hy)
        Ix = cv2.filter2D(B, -1, hx)
        gradmag = np.sqrt(Ix**2 + Iy**2)
        L = M

        fgm = cv2.erode(C, np.ones((7, 7), np.uint8))
        gradmagp = cv2.bitwise_not(cv2.bitwise_or(gradmag, cv2.bitwise_or(bgm, fgm)))
        F = cv2.connectedComponentsWithStats(cv2.watershed(gradmagp), connectivity=8)[1]
        L[F == 0] = 0

    return L
