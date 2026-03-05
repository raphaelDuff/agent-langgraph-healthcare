from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from agent.utils.state import PrescriptionGraphState
from agent.utils.nodes import (
    extract_drug_prescription,
    has_prescription,
    route_if_prescription,
)


def build_graph() -> CompiledStateGraph:

    agent_builder = StateGraph(PrescriptionGraphState)
    agent_builder.add_node("has_prescription_node", has_prescription)
    agent_builder.add_node("extract_drug_prescription_node", extract_drug_prescription)
    agent_builder.add_edge(START, "has_prescription_node")
    agent_builder.add_conditional_edges(
        "has_prescription_node",
        route_if_prescription,
        {
            "extract_drug_prescription_node": "extract_drug_prescription_node",
            "END": END,
        },
    )

    # TODO: Verify if we will need MemorySaver()
    return agent_builder.compile()


# chain_image = chain.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(chain_image)

# test_transcript = "Doctor: I'm prescribing amoxicillin 4000mg. Patient: How often should I take it? Doctor: Three times daily for 7 days."


# async def main():
#     state = await chain.ainvoke(PrescriptionGraphState(transcript=test_transcript))
#     print("Has prescription:", state["has_prescription_output"])
#     print("Extracted:", state["extracted_prescriptions"])


# asyncio.run(main())
