from operator import itemgetter
from pydantic import BaseModel

from langchain_community.llms import Replicate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub as prompts

from medcat.cat import CAT

from .config import config


class ExtractorModel(BaseModel):
    model: CAT

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        # Convert the custom class instance to a dictionary
        custom_instance_dict = {
            "cat": str(self.model),
        }

        # Call the dict() method of the parent class to get the Pydantic model as a dictionary
        base_dict = super().dict(*args, **kwargs)

        # Include the custom class dictionary in the final dictionary
        base_dict["model"] = custom_instance_dict

        return base_dict


def extract_entities(_dict, context):
    cat = _dict["model"].model
    results = cat.get_entities(_dict["text"])
    context["extracted_entities"] = {
        item["source_value"]: {
            "cui": item["cui"],
            "start": item["start"],
            "end": item["end"],
        }
        for item in results["entities"].values()
    }
    return [item["source_value"] for item in results["entities"].values()]


def update_with_cuis(llm_output, context):
    concept_dict = context["extracted_entities"]
    for item in llm_output["values"]:
        node_1_id = concept_dict.get(item["node_1"])
        node_2_id = concept_dict.get(item["node_2"])
        if node_1_id:
            item["node_1"] = item["node_1"] + " | " + node_1_id.get("cui", "")
        if node_2_id:
            item["node_2"] = item["node_2"] + " | " + node_2_id.get("cui", "")
    llm_output["context"] = context

    return llm_output


# LLM
replicate_id = config.replicate_id
model = Replicate(
    model=replicate_id,
    model_kwargs={"temperature": config.temperature},
)

medcat_path = config.medcat_model_path
cat = CAT.load_model_pack(medcat_path)

prompt = prompts.pull(config.prompt_id)

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

# Context dictionary to store memory
context = {"extracted_entities": {}}

# Chain
chain = (
    {
        "note": RunnablePassthrough(),
        "output_schema": lambda x: json_schema,
        "concepts": {
            "text": itemgetter("note"),
            "model": lambda x: ExtractorModel(model=cat),
        }
        | RunnableLambda(lambda x: extract_entities(x, context)),
    }
    | prompt
    | model
    | JsonOutputParser()
    | RunnableLambda(lambda x: update_with_cuis(x, context))
)
