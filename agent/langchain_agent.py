from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from typing import Optional
import json


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
    confidence: float


class PrescriptionExtractionResult(BaseModel):
    prescriptions: list[ExtractedPrescription]
    missing_fields: list[str]
    extraction_confidence: float


@tool
async def get_drug_prescription(transcript: str) -> ExtractedPrescription:
    """Analyze a medical transcript to extract structured information.

    Args:
        transcript: raw text of doctor-patient conversation
    """
    system_prompt = """You are a medical AI assistant specializing in analyzing doctor-patient conversations.
Your task is to:
1. Identify speakers and their roles (doctor vs patient)
2. Extract prescription information in structured format
3. Generate clarifying questions for missing CRITICAL safety information
4. Provide a summary of the conversation

CRITICAL CLARIFYING QUESTIONS - If any of the following information is missing from the transcript, you MUST ask about it:
- Patient allergies (MANDATORY if not mentioned)
- Current medications for interaction checking (MANDATORY if not mentioned)
- Patient weight/age for dosage calculation (MANDATORY if not mentioned)
- Pregnancy status (if relevant to the prescribed medication)
- Kidney/liver function (if relevant for the drug metabolism)
- Any ambiguous prescription details (dosage, frequency, duration)

IMPORTANT: Return your response as valid JSON only, with this exact structure:
{
    "speaker_roles": {"Speaker Name": "role"},
    "prescriptions": [
        {
            "drug_name": "string",
            "dosage": "string",
            "dosage_unit": "string",
            "dosage_value": number,
            "form": "string",
            "frequency": "string",
            "duration": "string or null",
            "route": "string",
            "instructions": "string or null",
            "confidence": number (0.0-1.0)
        }
    ],
    "clarifying_questions": ["question1", "question2"],
    "summary": "string",
    "confidence": number (0.0-1.0)
}"""

    user_prompt = f"""Analyze this medical transcript and extract the required information:

TRANSCRIPT:
{transcript}

Remember to return ONLY valid JSON with the exact structure specified."""

    try:
        client = OpenAI(api_key=api_key)
        model = "open-ai-model-name"
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response
        response_text = response.content[0].text
        clean_json = self._extract_json_from_response(response_text)

        # Parse JSON response
        analysis_data = json.loads(clean_json)

        # Convert prescriptions to ExtractedPrescriptionModel
        prescriptions = [
            ExtractedPrescriptionModel(
                drug_name=p["drug_name"],
                dosage=p["dosage"],
                dosage_unit=p["dosage_unit"],
                dosage_value=float(p["dosage_value"]),
                form=p.get("form") or "tablet",
                frequency=p["frequency"],
                duration=p.get("duration"),
                route=p.get("route") or "oral",
                instructions=p.get("instructions"),
                confidence=float(p.get("confidence", 0.8)),
            )
            for p in analysis_data.get("prescriptions", [])
        ]

        return TranscriptAnalysisResult(
            speaker_roles=analysis_data.get("speaker_roles", {}),
            prescriptions=prescriptions,
            clarifying_questions=analysis_data.get("clarifying_questions", []),
            summary=analysis_data.get("summary", "Analysis completed"),
            confidence=float(analysis_data.get("confidence", 0.8)),
        )

    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        # Log the raw response for debugging
        print(f"JSON parsing error: {str(e)}")
        print(f"Raw response: {response_text[:500]}")  # First 500 chars
        return TranscriptAnalysisResult(
            speaker_roles={"Doctor": "doctor", "Patient": "patient"},
            prescriptions=[],
            clarifying_questions=[
                "Could you please rephrase the prescription information more clearly?"
            ],
            summary=f"Failed to parse transcript analysis: {str(e)}. Raw response: {response_text[:100]}",
            confidence=0.3,
        )
    except Exception as e:
        raise ValueError(f"Failed to analyze transcript with Claude: {str(e)}")
