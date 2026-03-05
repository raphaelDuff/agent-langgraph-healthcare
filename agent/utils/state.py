from pydantic import BaseModel
from typing import Optional
from agent.utils.models import HasPrescriptionModel, ListPrescriptionExtractionModel


class PrescriptionGraphState(BaseModel):
    llm_calls: int = 0
    transcript: Optional[str] = None
    extracted_prescriptions: Optional[ListPrescriptionExtractionModel] = None
    summary: Optional[str] = None
    has_prescription_output: Optional[HasPrescriptionModel] = None
