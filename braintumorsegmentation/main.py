import os
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import model
import utils

from tests import dummy_patients_test
from models import Patient, Queue, PatientData
import matplotlib.image
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
    sorted_predictions = sorted(list(queue.patients.values()), key=lambda x: x.danger, reverse=True)
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


@app.get('/images/{patient_id}')
def get_patient_data(patient_id: str):
    p = find_by_uuid(patient_id)
    if p is not None:
        image = model.read_3d(patient_id)
        
        for version in image:
            for i, slice in enumerate(version):
                matplotlib.image.imsave(os.path.join("tmp", f"image_{patient_id}_{i}.png"), slice)

        tumor_map=utils.read_prediction(patient_id)

        for i, slice in enumerate(tumor_map):
                matplotlib.image.imsave(os.path.join("tmp", f"mask_{patient_id}_{i}.png"), slice)
        return {"length" : len(image[0])}
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


if __name__ == "__main__":
    for folder in ['tmp', 'input', "no_skull", "tests", "registered", "predictions"]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    '''
    patients = dummy_patients_test.get_patients()# trwa długo - testuje też ładowanie predykcji
    for patient in patients:
        queue.patients[patient.id] = patient
    '''
    patient_names = ['A', 'B', 'C']
    for patient_name, patient_id in zip(patient_names, os.listdir('no_skull')):
        queue.patients[patient_id] = patient = Patient(
            id=patient_id,
            name=patient_name,
            link='https://example.com',
            danger=int(np.sum(utils.read_prediction(patient_id))),
        )
    
    uvicorn.run(app, host="10.186.0.9", port=8000)
