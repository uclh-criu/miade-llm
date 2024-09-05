import yaml
from operator import itemgetter
from pydantic import BaseModel

from langchain_community.llms import Replicate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub

from medcat.cat import CAT

class ExtractorModel(BaseModel):
    model: CAT
    
    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        # Convert the custom class instance to a dictionary
        custom_instance_dict = {
            'cat': str(self.model),
        }

        # Call the dict() method of the parent class to get the Pydantic model as a dictionary
        base_dict = super().dict(*args, **kwargs)

        # Include the custom class dictionary in the final dictionary
        base_dict['model'] = custom_instance_dict

        return base_dict


def extract_entities(_dict):
    cat = _dict['model'].model
    results = cat.get_entities(_dict['text'])
    return [item['source_value'] for item in results['entities'].values()]


# Load configuration from YAML file
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# LLM
replicate_id = config.get("replicate_id", "mistralai/mixtral-8x7b-instruct-v0.1:5d78bcd7a992c4b793465bcdcf551dc2ab9668d12bb7aa714557a21c1e77041c") # noqa: E501
model = Replicate(
    model=replicate_id,
    model_kwargs={"temperature": 0},
)

medcat_path = config.get("medcat_model_path", "../data/models/miade_mimic_problems_unsupervised_trained_modelpack_w_meta_jun_2023_f25ec9423958e8d6.zip")
cat = CAT.load_model_pack(medcat_path)

prompt = hub.pull(config.get("prompt_id", "jenniferjiang/extract-medical-entity-relations-base"))

json_schema = """{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Medical entities and relations in a medical discharge summary",
    "description": "Clinically significant medical entities and the relationships between them",
    "type": "object",
    "properties": {
        "values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_1": {"title": "Node_1", "description": "A medical entity found in the document", "type": "string"},
                    "node_2": {"title": "Node_2", "description": "Another medical entity found in the document", "type": "string"},
                    "edge": {"title": "Edge", "description": "The clinical relation between entities node_1 and node_2 in a few words from the note", "type": "string"}
                },
                "required": ["node_1", "node_2", "edge"]
            }
        }
    },
    "required": ["values"]
}
"""
# Chain
chain = (
    {
        "note": RunnablePassthrough(),
        "output_schema": lambda x: json_schema,
        "concepts": {"text": itemgetter("note"), "model": lambda x: ExtractorModel(model=cat)} | RunnableLambda(extract_entities)
    } 
    | prompt 
    | model
    | JsonOutputParser()
)
