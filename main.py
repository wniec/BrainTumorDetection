import os.path
import h5py
import numpy as np
from matplotlib import pyplot as plt

import dicom_transformer
import transform
import model

if __name__ == "__main__":
    patient_name = input("enter patient name: ")
    transform.register(patient_name)
    transform.bet_transform(patient_name)
    prediction = model.prediction_for_volume(patient_name)

    pred_path = os.path.join("predictions", patient_name)
    os.mkdir(pred_path)
    with h5py.File(os.path.join(pred_path, 'prediction.h5'), "w") as f:
        f.create_dataset("prediction", data=prediction)
    priority_value = np.sum(prediction)
    print(f"priority value is {priority_value:.2f}")
    '''
    img = dicom_transformer.load_dicom('T1nii/image.0001.dcm')
    plt.imshow(img.pixel_array)
    plt.show()
    dicom_transformer.dicom_to_nii_gz(img, 'output.nii.gz')
    '''
