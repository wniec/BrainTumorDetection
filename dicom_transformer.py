import nibabel as nib
import pydicom
import numpy as np
import SimpleITK as sitk


def load_dicom(dicom_file):
    return pydicom.dcmread(dicom_file)


def dicom_to_nii_gz(dcm, filename):
    data = dcm.pixel_array
    new_image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(new_image, filename=filename)


def nii_gz_to_dicom(in_file: str):
    array = (nib.load(in_file).get_fdata() * 255).astype(np.uint8)
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, "nii2dcm.dcm")
