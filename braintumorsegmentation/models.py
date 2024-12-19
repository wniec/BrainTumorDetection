from typing import Dict, List, Optional
from datetime import date

from pydantic import BaseModel


class PacientScanData(BaseModel):
    id: str
    name: str
    danger: int
    priority: int
    scan_date: Optional[str] = None


class InternalPatient(BaseModel):
    id: str
    name: str
    danger: int
    priority: int
    scan_date: Optional[date] = None


class PatientData(BaseModel):
    patient: PacientScanData
    image: List[List[List[List[float]]]]
    tumor_map: List[List[List[float]]]


class Queue(BaseModel):
    patients: Dict[str, InternalPatient] = dict()
