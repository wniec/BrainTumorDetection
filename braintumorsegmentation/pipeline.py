import os.path
import h5py
import transform
import model
import utils
from models import Patient


def transform_predict(patient: Patient, tests=False):
    transform.register(patient.id)
    if tests is False:
        transform.bet_transform(patient.id)
    else:
        for f in os.listdir("registered"):
            src_path = os.path.join("registered", f)
            dst_path = os.path.join("no_skull", f)
            os.rename(src_path, dst_path)
    prediction = model.prediction_for_volume(patient.id)
    with h5py.File(os.path.join("predictions", f"{patient.id}.h5"), "w") as f:
        f.create_dataset("prediction", data=prediction)
    priority_value = utils.get_danger(patient.id)
    print(f"priority value of {patient.name} is {priority_value:.2f}")
    return priority_value

    # img = dicom_transformer.load_dicom('nii2dcm.dcm')
    # plt.imshow(img.pixel_array[:, :, 80])
    # plt.show()
    # dicom_transformer.dicom_to_nii_gz(img, 'output.nii.gz')
    # in_file = os.path.join('input', "T2.nii.gz")
    # dicom_transformer.nii_gz_to_dicom(in_file)
