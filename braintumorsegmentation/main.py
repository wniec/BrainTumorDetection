import io
import os
import socket
import zipfile
from typing import List

import uvicorn
from starlette.responses import Response

from braintumorsegmentation.models import InternalPatient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from imageio import v3 as iio
from fastapi.responses import StreamingResponse

import utils
import dummy_patients_test
from models import InternalPatient, PacientScanData, Queue
from db_utils import db_conn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = Queue()


@app.get("/patients", response_model=List[PacientScanData])
def general_predictions():
    with db_conn() as db:
        db_res = db.get_top_not_completed_pacients_ordered()
    returnlist = [map_patient(pacient) for pacient in db_res]
    print("Returning: ", returnlist)
    return returnlist


def find_by_uuid(patient_id: str):
    if patient_id in queue.patients:
        return queue.patients[patient_id]
    else:
        return None


def map_patient(pacient: InternalPatient) -> PacientScanData:
    return PacientScanData(
        id=pacient.id,
        name=pacient.name,
        danger=int(pacient.danger * 10000),
        priority=int(pacient.danger * 10000),
        scan_date=None,
    )


@app.get("/pred/{patient_id}", response_model=PacientScanData)
def specific_patients(patient_id: str):
    p = find_by_uuid(patient_id)
    if p is not None:
        return p
    else:
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/images/{patient_id}")
def get_patient_data_full_head(patient_id: str):
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


@app.post("/delete_list")
def delete_from_list(data: dict):
    # print(data)
    with db_conn() as db:
        db.set_pacient_completed(data["patiend_id"], data["erase"])


@app.get("/reset")
def delete_from_list():
    with db_conn() as db:
        db.uncomplete_all()
    return "RESET!"


if __name__ == "__main__":
    """
    for folder in ["input", "no_skull", "tests", "registered", "predictions"]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    patients = (
        dummy_patients_test.get_patients()
    )  # trwa długo - testuje też ładowanie predykcji
    """
    patients = []
    patient_names = ["Alice", "Bob", "Carol", "Dave", "Eva"]
    for patient_name, patient_id in zip(patient_names, os.listdir("no_skull")):
        if os.path.exists(os.path.join("no_skull", patient_id, "FLAIR.nii.gz")):
            print(f'patient {patient_name} has FLAIR image done')
        danger = utils.get_danger(patient_id)
        patients.append(InternalPatient(
            id=patient_id,
            name=patient_name,
            link="https://example.com",
            danger=danger,
            priority=danger
        ))
    # """

    with db_conn() as db:
        db.clear_all_data()
        for patient in patients:
            db.add_pacient(patient.id, patient.name)
            db.add_imaging(patient.id, patient.danger)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    hostname = "localhost"
    s.close()
    uvicorn.run(app, host=hostname, port=8000)
