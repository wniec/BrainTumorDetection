import h5py
import nibabel as nib
import os
import numpy as np


def read_3d(patient_id: str) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    flair_path = os.path.join(path, "FLAIR.nii.gz")
    t1 = nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(path, "T2.nii.gz")).get_fdata()
    if os.path.exists(flair_path):
        flair = nib.load(flair_path).get_fdata()
        image = np.zeros((3, 155, 240, 240))
        image[0, :, :, :] = np.array(t1).transpose(2, 0, 1)
        image[1, :, :, :] = np.array(t2).transpose(2, 0, 1)
        image[2, :, :, :] = np.array(flair).transpose(2, 0, 1)
        return image
    else:
        image = np.zeros((2, 155, 240, 240))
        image[0, :, :, :] = np.array(t1).transpose(2, 0, 1)
        image[1, :, :, :] = np.array(t2).transpose(2, 0, 1)
        return image


def read_2d(patient_id: str, index: int, mode: str) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    if mode == "prediction":
        tumor_map = read_prediction(patient_id)
        image = np.zeros((240, 240, 4))
        danger = np.array(tumor_map)[index, :, :]
        image[:, :, 0] = np.ones((240, 240))
        image[:, :, 3] = danger
        pass
    elif mode != "profile":
        background = nib.load(os.path.join(path, f"{mode.upper()}.nii.gz")).get_fdata()[
                     :, :, index
                     ]
        background = background / (np.max(background) if np.max(background) > 0 else 1)
        image = np.repeat(background[:, :, np.newaxis], 3, axis=2)
    else:
        t1 = (
            nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()[120, :, :].transpose()
        )
        t1 = t1 / (np.max(t1) if np.max(t1) > 0 else 1)
        image = np.repeat(t1[:, :, np.newaxis], 3, axis=2)
        image[index, :, 1] = np.ones((1, 240))
        image = image[::-1, :]
    return (image * 255).astype(np.uint8)


def read_prediction(patient_id: str):
    path = os.path.join("predictions", f"{patient_id}.h5")
    with h5py.File(path, "r") as f:
        data = np.array(f["prediction"])
    return data


def get_brain_volume_for_pacient(patient_id: str) -> float:
    path = os.path.join("no_skull", patient_id)
    t1 = nib.load(os.path.join(path, "T1.nii.gz"))
    return get_brain_volume(t1)


def get_brain_volume(nii_img: nib.nifti1.Nifti1Image):
    return get_brain_percentage(nii_img.get_fdata()) * get_image_volume(nii_img)


def get_brain_percentage(img: np.array):
    return 1 - len(img[img == 0]) / len(img[img >= 0])


def get_image_volume(nii_img: nib.nifti1.Nifti1Image):
    dim = nii_img.header["dim"]
    return np.prod(dim[1:dim[0] + 1])


def get_danger(patient_id: str) -> int:
    return int(np.sum(read_prediction(patient_id)) / get_brain_volume_for_pacient(patient_id) * 10000)
