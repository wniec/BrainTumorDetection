import os.path
import h5py
import numpy as np
import transform
import model
from models import Patient


def transform_predict(patient: Patient):
    transform.register(patient.id)
    transform.bet_transform(patient.id)
    prediction = model.prediction_for_volume(patient.id)
    with h5py.File(os.path.join("predictions", f'{patient.id}.h5'), "w") as f:
        f.create_dataset("prediction", data=prediction)
    priority_value = np.sum(prediction)
    print(f"priority value of {patient.name} is {priority_value:.2f}")
    return priority_value

    # img = dicom_transformer.load_dicom('nii2dcm.dcm')
    # plt.imshow(img.pixel_array[:, :, 80])
    # plt.show()
    # dicom_transformer.dicom_to_nii_gz(img, 'output.nii.gz')
    # in_file = os.path.join('input', "T2.nii.gz")
    # dicom_transformer.nii_gz_to_dicom(in_file)
