from typing import List
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


models = [("gpt-4o-2024-08-06", "openai"), ("claude-3-5-sonnet-20241022", "anthropic")]


class Concept(BaseModel):
    code: str
    name: str


class ConceptList(BaseModel):
    problems: List[Concept]
    medications: List[Concept]
    allergies: List[Concept]


raw_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "You are an expert at clinical structured data extraction. You use the SNOMED-CT coding system. You will be given unstructured text from a medical note and should convert it into the given structure.",
        },
        {"role": "user", "content": "{note}"},
    ]
)

model = init_chat_model(
    "gpt-4o-2024-08-06", model_provider="openai"
).with_structured_output(ConceptList)

chain = {"note": RunnablePassthrough()} | raw_prompt | model

chains = [
    (
        provider + "-" + model_name,
        {"note": RunnablePassthrough()}
        | raw_prompt
        | init_chat_model(model_name, model_provider=provider).with_structured_output(
            ConceptList
        ),
    )
    for model_name, provider in models
]
