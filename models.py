from typing import Dict

from pydantic import BaseModel


class Patient(BaseModel):
    id: str
    link: str
    name: str
    danger: int


class Queue(BaseModel):
    patients: Dict[str, Patient] = dict()
