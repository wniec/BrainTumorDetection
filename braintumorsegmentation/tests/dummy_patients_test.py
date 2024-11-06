import os
import shutil
import sys
import uuid

import pipeline
from models import Patient


def get_patients():
    patients = []
    for patient_name in ("A", "B", "C", "D"):
        patient = Patient(
            id=str(uuid.uuid4()),
            name=patient_name,
            link="https://example.com",
            danger=0,
        )
        shutil.copytree(
            os.path.join(os.path.dirname(sys.modules[__name__].__file__), patient_name),
            os.path.join("input", patient.id),
        )
        patient.danger = int(pipeline.transform_predict(patient, tests=True))
        patients.append(patient)
    return patients
