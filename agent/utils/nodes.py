from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from agent.utils.state import PrescriptionGraphState
from agent.utils.models import HasPrescriptionModel, ListPrescriptionExtractionModel

load_dotenv()


llm_key = os.getenv("OPENAI_APIKEY")

client = AsyncOpenAI(api_key=llm_key)


async def has_prescription(
    state: PrescriptionGraphState,
) -> dict:
    """From the message input, LLM verify if there is any drug prescription"""
    system_prompt = (
        "Determine whether the following doctor-patient conversation includes "
        "any medication prescription."
    )
    if state.transcript is None:
        raise ValueError("Failed to load transcript @ has_precription node")

    response = await client.responses.parse(
        model="gpt-4.1-mini",
        instructions=system_prompt,
        input=state.transcript,
        text_format=HasPrescriptionModel,
        temperature=0.2,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Failed to parse output answer related to has_precription node"
        )

    return {
        "has_prescription_output": HasPrescriptionModel.model_validate(
            response.output_parsed
        )
    }


async def route_if_prescription(state: PrescriptionGraphState) -> str:
    if state.has_prescription_output is None:
        raise ValueError("has_prescription_output missing")
    if state.has_prescription_output.has_prescription:
        return "extract_drug_prescription_node"
    return "END"


async def extract_drug_prescription(
    state: PrescriptionGraphState,
) -> dict:
    """
    Extract drug prescription information explicitly stated
    in a doctor-patient conversation.
    """

    system_prompt = """
You are a medical information extraction system.

TASK:
- Extract ONLY drug prescription information that is EXPLICITLY stated.
- Do NOT infer, assume, or complete missing information.
- If a field is not clearly stated, return null.
- Do NOT generate questions.
- Do NOT summarize.
- Do NOT evaluate safety.

Return structured data only.
"""

    if not isinstance(state.transcript, str):
        raise ValueError(
            "Failed to read transcript_text on extract_drug_prescription node"
        )

    response = await client.responses.parse(
        model="gpt-4.1-mini",
        instructions=system_prompt,
        input=state.transcript,
        text_format=ListPrescriptionExtractionModel,
        temperature=0,
    )

    if response.output_parsed is None:
        raise ValueError("Failed to parse prescription extraction output")
    return {
        "extracted_prescriptions": ListPrescriptionExtractionModel.model_validate(
            response.output_parsed
        )
    }
