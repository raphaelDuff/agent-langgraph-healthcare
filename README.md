# Medical Transcript AI Agent (LangGraph)

This repository demonstrates a production-oriented AI system for analyzing doctor–patient conversations using LangChain and LangGraph.

The system processes raw medical transcripts, determines whether a drug prescription is present, extracts structured prescription data using LLM-powered tools, validates the safety of the transcription, and returns auditable results. The architecture emphasizes clear separation between agents and tools to improve reliability, safety, and maintainability.

---

## Overview

The project is designed around a graph-based execution model where:

- A routing agent controls decision-making and flow
- LLM-powered tools perform constrained, single-purpose transformations
- Safety validation is isolated and explainable
- Outputs are structured and suitable for downstream systems

This approach avoids the common anti-pattern of using a single autonomous agent for all tasks.

---

## High-Level Architecture

```text
[ START ]
   |
   v
[ Router Agent ]
   |
   |-- No prescription --> [ Conversation Summary ] --> [ END ]
   |
   |-- Prescription --> [ Prescription Extraction Tool ]
                             |
                             v
                      [ Safety Validation Tool ]
                             |
                             v
                      [ Response Assembler ]
                             |
                           [ END ]
