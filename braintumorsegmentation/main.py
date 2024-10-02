import io
import os
from typing import List
from imageio import v3 as iio
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import utils

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


@app.get("/images/{patient_id}/{image_id}/{mode}")
def get_patient_data_t1(patient_id: str, image_id: int, mode: str):
    if not (mode == "t1" or mode == "t2" or mode == "profile"):
        raise HTTPException(
            status_code=400, detail="selected image mode is not supported"
        )
    p = find_by_uuid(patient_id)
    if p is not None:
        image = utils.read_2d(patient_id, image_id, mode)
        with io.BytesIO() as buf:
            iio.imwrite(buf, image, plugin="pillow", format="PNG")
            im_bytes = buf.getvalue()
        headers = {"Content-Disposition": 'inline; filename="test.png"'}
        return Response(im_bytes, headers=headers, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


if __name__ == "__main__":
    for folder in ['tmp', 'input', "no_skull", "tests", "registered", "predictions"]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    patients = dummy_patients_test.get_patients()# trwa długo - testuje też ładowanie predykcji
    for patient in patients:
        queue.patients[patient.id] = patient
    patient_names = ["A", "B", "C"]
    for patient_name, patient_id in zip(patient_names, os.listdir("no_skull")):
        queue.patients[patient_id] = patient = Patient(
            id=patient_id,
            name=patient_name,
            link="https://example.com",
            danger=int(np.sum(utils.read_prediction(patient_id))),
        )

    uvicorn.run(app, host="localhost", port=8000)
