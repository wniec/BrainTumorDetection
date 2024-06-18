import h5py
import nibabel as nib
import os
import numpy as np


def read_3d(patient_id: str):
    path = os.path.join("no_skull", patient_id)
    t1 = nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(path, "T2.nii.gz")).get_fdata()
    image = np.zeros((2, 155, 240, 240))
    image[0, :, :, :] = np.array(t1).transpose(2, 0, 1)
    image[1, :, :, :] = np.array(t2).transpose(2, 0, 1)
    # os.rmdir(os.path.join(path, "T1.nii.gz"))
    # os.rmdir(os.path.join(path, "T2.nii.gz"))
    return image


def read_prediction(patient_id: str):
    path = os.path.join("predictions", f'{patient_id}.h5')
    with h5py.File(path, 'r') as f:
        data = np.array(f['prediction'])
    return data
