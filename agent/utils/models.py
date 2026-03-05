from pydantic import BaseModel, Field
from typing import Optional


class ExtractedPrescription(BaseModel):
    drug_name: str
    dosage_value: Optional[float]
    dosage_unit: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    route: Optional[str]
    form: Optional[str]
    instructions: Optional[str]
    confidence: float = Field(gt=0, le=1.0)


class ListPrescriptionExtractionModel(BaseModel):
    prescriptions: list[ExtractedPrescription]
    missing_fields: list[str]
    extraction_confidence: float


class HasPrescriptionModel(BaseModel):
    has_prescription: Optional[bool] = None
    has_prescription_confidence: Optional[float] = Field(
        default=None,
        gt=0,
        le=1.0,
    )
