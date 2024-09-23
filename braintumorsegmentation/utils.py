import h5py
import nibabel as nib
import os
import numpy as np


def read_3d(patient_id: str) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    t1 = nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(path, "T2.nii.gz")).get_fdata()
    image = np.zeros((2, 155, 240, 240))
    image[0, :, :, :] = np.array(t1).transpose(2, 0, 1)
    image[1, :, :, :] = np.array(t2).transpose(2, 0, 1)
    return image


def read_2d(patient_id: str, index: int) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    t1 = nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(path, "T2.nii.gz")).get_fdata()
    tumor_map = read_prediction(patient_id)
    image = np.zeros((240, 240, 3))
    background = (np.array(t1)[:, :, index] + np.array(t2)[:, :, index]) / 2
    danger = np.array(tumor_map)[index, :, :]
    image[:, :, 2] = background
    image[:, :, 1] = np.where(np.abs(danger - 0.5) > 0.1, background, 1)
    image[:, :, 0] = np.where(danger < 0.4, background, 1)
    return (image * 255).astype(np.uint8)


def read_prediction(patient_id: str):
    path = os.path.join("predictions", f'{patient_id}.h5')
    with h5py.File(path, 'r') as f:
        data = np.array(f['prediction'])
    return data
