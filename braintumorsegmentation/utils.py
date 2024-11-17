from itertools import product
import h5py
import nibabel as nib
import os
import numpy as np
from PIL import Image


def read_3d(patient_id: str) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    flair_path = os.path.join(path, "FLAIR.nii.gz")
    t1 = nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(path, "T2.nii.gz")).get_fdata()
    image_data = [t1, t2]
    if os.path.exists(flair_path):
        flair = nib.load(flair_path).get_fdata()
        image_data.append(flair)
    image = np.stack([np.array(i).transpose(2, 0, 1) for i in image_data])
    if image.shape[2] != 240 or image.shape[3] != 240:
        image_resized = np.zeros((*image.shape[:2], 240, 240))
        for i, j in product(range(image.shape[0]), range(image.shape[1])):
            image_resized[i, j, :, :] = np.array(
                Image.fromarray(image[i, j, :, :]).resize(size=(240, 240))
            )
        image = image_resized
    if image.shape[1] != 155:
        image_resized = np.zeros((image.shape[0], 155, 240, 240))
        for i, j in product(range(image.shape[0]), range(240)):
            image_resized[i, :, j, :] = np.array(
                Image.fromarray(image[i, :, j, :]).resize(size=(240, 155))
            )
        image = image_resized
    return image


def read_2d(patient_id: str, index: int, mode: str) -> np.ndarray:
    path = os.path.join("no_skull", patient_id)
    if mode == "prediction":
        tumor_map = read_prediction(patient_id)
        image = np.zeros((240, 240, 4))
        danger = np.array(tumor_map)[index, :, :]
        image[:, :, 0] = np.ones((240, 240))
        image[:, :, 3] = danger
    elif mode != "profile":
        background = nib.load(os.path.join(path, f"{mode.upper()}.nii.gz")).get_fdata()
        index = int(index * (background.shape[2] - 1) / 154)
        background = background[:, :, index]
        background = np.array(
            Image.fromarray(
                background / (np.max(background) if np.max(background) > 0 else 1)
            ).resize(size=(240, 240))
        )
        image = np.repeat(background[:, :, np.newaxis], 3, axis=2)
    else:
        t1 = (
            nib.load(os.path.join(path, "T1.nii.gz")).get_fdata()[120, :, :].transpose()
        )
        t1 = np.array(
            Image.fromarray(t1 / (np.max(t1) if np.max(t1) > 0 else 1)).resize(
                size=(240, 155)
            )
        )
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
    return np.prod(dim[1 : dim[0] + 1])


def get_danger(patient_id: str) -> int:
    return int(
        np.sum(read_prediction(patient_id))
        / get_brain_volume_for_pacient(patient_id)
        * 10000
    )
