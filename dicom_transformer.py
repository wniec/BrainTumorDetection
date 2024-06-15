import nibabel as nib
import pydicom
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt


def load_dicom(dicom_file):
    return pydicom.dcmread(dicom_file)


def dicom_to_nii_gz(dcm, filename):
    data = dcm.pixel_array
    new_image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(new_image, filename=filename)
