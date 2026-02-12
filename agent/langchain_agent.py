from langchain.tools import tool
from openai import AsyncOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END


load_dotenv()
llm_key = os.getenv("OPENAI_APIKEY")


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


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


class PrescriptionGraphState(BaseModel):
    messages: list[AnyMessage]
    llm_calls: int = 0
    transcript: Optional[str] = None
    extracted_prescriptions: Optional[ListPrescriptionExtractionModel] = None
    summary: Optional[str] = None
    has_prescription_output: Optional[HasPrescriptionModel] = None


client = AsyncOpenAI(api_key=llm_key)


async def has_prescription(
    state: PrescriptionGraphState,
) -> dict:
    """From the message input, LLM verify if there is any drug prescription"""
    system_prompt = (
        "Determine whether the following doctor-patient conversation includes "
        "any medication prescription."
    )
    messages = state.messages
    last_message = messages[-1]
    if not isinstance(last_message.content, str):
        raise ValueError(
            "Failed to parse output answer related to has_precription node"
        )
    state.transcript = last_message.content

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


def route_if_prescription(state: PrescriptionGraphState) -> str:
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
        "prescription_output": ListPrescriptionExtractionModel.model_validate(
            response.output_parsed
        )
    }


agent_builder = StateGraph(PrescriptionGraphState)
agent_builder.add_node("has_prescription_node", has_prescription)
agent_builder.add_node("extract_drug_prescription_node", extract_drug_prescription)

agent_builder.add_edge(START, "has_prescription_node")
agent_builder.add_conditional_edges(
    "has_prescription_node",
    route_if_prescription,
    {"extract_drug_prescription": "extract_drug_prescription_node", "END": END},
)
