from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tests import dummy_patients_test
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


@app.get("/preds", response_model=List[Patient])
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


@app.get("/pred/{pred_id}", response_model=Patient)
def specific_patients(patient_id: str):
    p = find_by_uuid(patient_id)
    if p is not None:
        return p
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


if __name__ == "__main__":

    patients = dummy_patients_test.get_patients()# trwa długo - testuje też ładowanie predykcji

    for patient in patients:
        queue.patients[patient.id] = patient

    uvicorn.run(app, host="127.0.0.1", port=8000)
