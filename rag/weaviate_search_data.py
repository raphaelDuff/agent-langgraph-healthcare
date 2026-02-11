import weaviate
import json
from weaviate.classes.generate import GenerativeConfig


with weaviate.connect_to_local() as client:

    movies = client.collections.use("drug_dosage_collection")

    response = movies.generate.near_text(
        query="adrenalina",
        limit=1,
        grouped_task="Use only the dosage information from the object to determine the recommended dosage for children with anaphylaxis. Do not invent information.",
        generative_provider=GenerativeConfig.openai(
            model="gpt-4o-mini", max_tokens=500, temperature=0.2
        ),
    )

    for obj in response.objects:
        print(json.dumps(obj.properties, indent=2))  # Inspect the results


def retrieval_function(query: str) -> str:
    with weaviate.connect_to_local() as client:

        drugs = client.collections.use("drug_dosage_collection")

        response = drugs.generate.near_text(
            query="adrenalina",
            limit=1,
            grouped_task="Use only the dosage information from the object to determine the recommended dosage for children with anaphylaxis. Do not invent information.",
            generative_provider=GenerativeConfig.openai(
                model="gpt-4o-mini", max_tokens=500, temperature=0.2
            ),
        )
        if response.generative is not None:
            return f"Response: {response.generative.text}"

        return "Error: It was not found any recommended drug dosage."
