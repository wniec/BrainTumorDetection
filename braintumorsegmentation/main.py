import io
import os
import socket
import zipfile
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from imageio import v3 as iio
from starlette.responses import Response

import utils
from braintumorsegmentation.tests import dummy_patients_test
from models import Patient, Queue

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = Queue()


@app.get("/patients", response_model=List[Patient])
def general_predictions():
    sorted_predictions = sorted(
        list(queue.patients.values()), key=lambda x: x.danger, reverse=True
    )
    print("Returning: ", sorted_predictions)
    return sorted_predictions
    # return {"link": "Welcome to the REST API"}


def find_by_uuid(patient_id: str):
    if patient_id in queue.patients:
        return queue.patients[patient_id]
    else:
        return None


@app.get("/pred/{patient_id}", response_model=Patient)
def specific_patients(patient_id: str):
    p = find_by_uuid(patient_id)
    if p is not None:
        return p
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/images/{patient_id}")
def get_patient_data_full_head(patient_id: str):
    p = find_by_uuid(patient_id)
    if p is not None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for image_id in range(155):
                for mode in ["t1", "t2", "profile", "prediction"] + (
                    ["flair"]
                    if os.path.exists(
                        os.path.join("no_skull", patient_id, "FLAIR.nii.gz")
                    )
                    else []
                ):
                    im_buf = io.BytesIO()
                    image = utils.read_2d(patient_id, image_id, mode)
                    iio.imwrite(im_buf, image, plugin="pillow", format="PNG")
                    im_buf.seek(0)
                    zf.writestr(f"patient/{image_id}/{mode}.png", im_buf.read())

        buf.seek(0)

        headers = {"Content-Disposition": 'attachment; filename="test.zip"'}
        return Response(buf.getvalue(), media_type="application/zip", headers=headers)


if __name__ == "__main__":
    """
    for folder in ["tmp", "input", "no_skull", "tests", "registered", "predictions"]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    patients = (
        dummy_patients_test.get_patients()
    )  # trwa długo - testuje też ładowanie predykcji
    for patient in patients:
        queue.patients[patient.id] = patient
    """
    patient_names = ["A", "B", "C", "D"]
    for patient_name, patient_id in zip(patient_names, os.listdir("no_skull")):
        if os.path.exists(os.path.join("no_skull", patient_id, "FLAIR.nii.gz")):
            print(f'patient {patient_name} has FLAIR image done')
        queue.patients[patient_id] = patient = Patient(
            id=patient_id,
            name=patient_name,
            link="https://example.com",
            danger=int(np.sum(utils.read_prediction(patient_id))),
        )
    # """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    hostname = s.getsockname()[0]
    s.close()
    uvicorn.run(app, host=hostname, port=8000)
