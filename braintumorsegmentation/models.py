from typing import Dict, List

from pydantic import BaseModel


class Patient(BaseModel):
    id: str
    link: str
    name: str
    danger: int


class PatientData(BaseModel):
    patient: Patient
    image: List[List[List[List[float]]]] 
    tumor_map: List[List[List[float]]]


class Queue(BaseModel):
    patients: Dict[str, Patient] = dict()
