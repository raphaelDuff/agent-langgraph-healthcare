import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
import json
from tqdm import tqdm
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_json_data(file_path: str, base_dir=BASE_DIR) -> list[dict[str, str]]:
    with open(base_dir / file_path, "r", encoding="utf-8") as f:
        drugs = json.load(f)

    return drugs


# Step 1.1: Connect to your local Weaviate instance
with weaviate.connect_to_local() as client:

    if client.collections.exists("drug_dosage_collection"):
        client.collections.delete("drug_dosage_collection")

    # Step 1.2: Create a collection
    collection = client.collections.create(
        name="drug_dosage_collection",
        vector_config=Configure.Vectors.text2vec_ollama(  # Configure the Ollama embedding integration
            api_endpoint="http://ollama:11434",  # If using Docker you might need: http://host.docker.internal:11434
            model="nomic-embed-text",  # The model to use
        ),
        properties=[
            Property(
                name="drug",
                vectorize_property_name=True,
                data_type=DataType.TEXT,
                tokenization=Tokenization.LOWERCASE,
            ),
            Property(
                name="dosage", vectorize_property_name=True, data_type=DataType.TEXT
            ),
            Property(
                name="adverse_effects",
                vectorize_property_name=False,
                data_type=DataType.TEXT,
            ),
            Property(
                name="reference_link",
                vectorize_property_name=False,
                data_type=DataType.TEXT,
            ),
        ],
    )

    with collection.batch.fixed_size(batch_size=1, concurrent_requests=1) as batch:
        data_objects = load_json_data("sample_data.json")
        for obj in tqdm(data_objects):
            batch.add_object(properties=obj)

    print(
        f"Imported & vectorized {len(collection)} objects into the Drug Dosage Collection"
    )
