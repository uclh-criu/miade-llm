from typing import List, Optional
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


models = [
    ("gpt-4o-2024-08-06", "openai"),
    ("claude-3-5-sonnet-20241022", "anthropic"),
    ("mistral-large-latest", "mistralai"),
]


class Condition(BaseModel):
    name: str
    severity: Optional[str]
    bodySite: Optional[str]
    clinicalStatus: Optional[str]
    onset: Optional[str]
    abatement: Optional[str]
    stage: Optional[str]
    evidence: Optional[str]
    manifestation: Optional[str]
    cause: Optional[str]
    note: Optional[str]

class ConditionList(BaseModel):
    problems: List[Condition]


raw_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
            You are an expert at interpreting clinical notes and extracting information in structured formats conforming to FHIR resources. You will be given unstructured text from a medical note and should extract information about patient conditions in order to create a problem list, according to the following guidance from the FHIR documentation:

            You only include details mentioned in the note, you do not hallucinate, you do not make any inferences.

            The 'condition' resource is used to record detailed information about a condition, problem, diagnosis, or other event, situation, issue, or clinical concept that has risen to a level of concern. The condition could be a point in time diagnosis in context of an encounter, it could be an item on the practitioner’s Problem List, or it could be a concern that doesn’t exist on the practitioner’s Problem List. Oftentimes, a condition is about a clinician's assessment and assertion of a particular aspect of a patient's state of health. It can be used to record information about a disease/illness identified from application of clinical reasoning over the pathologic and pathophysiologic findings (diagnosis), or identification of health issues/situations that a practitioner considers harmful, potentially harmful and may be investigated and managed (problem), or other health issue/situation that may require ongoing monitoring and/or management (health issue/concern).

            The condition resource may be used to record a certain health state of a patient which does not normally present a negative outcome, e.g. pregnancy. The condition resource may be used to record a condition following a procedure, such as the condition of Amputee-BKA following an amputation procedure.

            While conditions are frequently a result of a clinician's assessment and assertion of a particular aspect of a patient's state of health, conditions can also be expressed by the patient, related person, or any care team member. A clinician may have a concern about a patient condition (e.g. anorexia) that the patient is not concerned about. Likewise, the patient may have a condition (e.g. hair loss) that does not rise to the level of importance such that it belongs on a practitioner’s Problem List. For example, each of the following conditions could rise to the level of importance such that it belongs on a problem or concern list due to its direct or indirect impact on the patient’s health:
            - Unemployed
            - Without transportation (or other barriers)
            - Susceptibility to falls
            - Exposure to communicable disease
            - Family History of cardiovascular disease
            - Fear of cancer
            - Cardiac pacemaker
            - Amputee-BKA
            - Risk of Zika virus following travel to a country
            - Former smoker
            - Travel to a country planned (that warrants immunizations)
            - Motor Vehicle Accident
            - Patient has had coronary bypass graft

            As per the FHIR condition resource, each condition should be structured in the following way. Include SNOMED CT term descriptions if possible but not SNOMED CT concept IDs. If there is no relevant information for a field, leave it blank.
            - name: Identification of the condition, problem or diagnosis (mandatory)
            - severity: Subjective severity of condition
            - bodySite: Anatomical location, if relevant
            - clinicalStatus: active | recurrence | relapse | inactive | remission | resolved | unknown
            - onset: Estimated or actual date, date-time, or age of onset
            - abatement: Estimated or actual date, date-time, or age when in resolution or remission
            - stage: Stage/grade, usually assessed formally
            - evidence: Supporting evidence for the condition
            - manifestation: Manifestations of the condition
            - note: Other information about the condition, in free text
            """,
        },
        {"role": "user", "content": "{note}"},
    ]
)

chains = [
    (
        provider + "-" + model_name,
        {"note": RunnablePassthrough()}
        | raw_prompt
        | init_chat_model(model_name, model_provider=provider).with_structured_output(
            ConditionList
        ),
    )
    for model_name, provider in models
]
