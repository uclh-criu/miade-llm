# miade-llm

MiADE supercharged with LLMs - for the detailed extraction of diagnosis.

This project uses LangServe to deploy langchain chains as REST API endpoints.

## Environment Setup

This project uses the [Replicate](https://replicate.com/) API to run models. You will need to set the `REPLICATE_API_TOKEN` environment variable (you usually get some free credits when you sign up). If you would like to use a different service or use self-hosted models, you would need to update the `relation_extractor` chain.

**Update**: The previous tested version of the `relation_extractor` chain used [Mixtral-8x7B-instruct-v0.1](https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1/versions), but this model is no longer available on Replicate. The current default model is [`microsoft/phi-3-mini-128k-instruct`](https://replicate.com/microsoft/phi-3-mini-128k-instruct).

Prompts are currently pulled from [LangChain Hub](https://smith.langchain.com/hub) so you also need to set `LANGCHAIN_API_KEY`. You can view the full prompt [here](https://smith.langchain.com/hub/jenniferjiang/extract-medical-entity-relations-base).

The model id, prompt, and extra model paths can be configured in `config/config.yaml`

(Optional) If you also want to configure [LangSmith](https://smith.langchain.com/) to trace and monitor chains, set these environment variables:

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Usage

To install dependencies make sure you have `poetry` installed:

```shell
pip install poetry
```

Then install the project dependencies with poetry:

```shell
cd src
poetry install
```

To spin up a LangServer instance run (make sure you are in the `src` directory):

```shell
poetry run langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all endpoints at [http://localhost:8000/docs](http://localhost:8000/docs).

Access the playground at [http://localhost:8000/name-of-package/playground](http://localhost:8000/name-of-package/playground)

Access the endpoints from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/name-of-package")
```
## Chains
### relation-extractor
Extracts relations between concepts found in note and outputs in a JSON-format. Uses MedCAT for NER (requires model).

MedCAT model is required to run this chain. To download an example model trained on MIMIC:

```bash
pip install gdown
gdown 'https://drive.google.com/uc?export=download&id=17s999FIotRenltR6gr_f8ZjdaXc-u1Gx', -O ./data/models/miade_problems_model_f25ec9423958e8d6.zip
```

## Experimental
### SNOMED RAG Agent
An experimental agent that maps clinical concepts to SNOMED CT codes using [LangGraph](https://langchain-ai.github.io/langgraph/). The agent:

1. Takes a clinical relation triplet (e.g., "fracture of left femur") as input
2. Generates appropriate SNOMED CT search terms
3. Queries a SNOMED CT terminology server
4. Predicts and evaluates morphology and finding site attributes
5. Scores candidate terms to find the best SNOMED CT concept match

Example usage can be found in `notebooks/snomed_agent.ipynb`. Requires access to a SNOMED CT terminology server. Currently uses the [Snowstorm SNOMED CT server](https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/swagger-ui/index.html#/).

#### Graph Components
##### Nodes
1. **Search**
- Takes a relation triplet (e.g., `{"node_1": "fracture", "node_2": "left femur", "edge": "of"}`)
- The `search` node uses GPT-4 to generate appropriate search variations (e.g., "left femur fracture", "fracture of left femur").
- These terms are used to query the SNOMED CT terminology server

2. **Planner**
- The `planner` node analyzes the primary search term
- Predicts two key SNOMED attributes:
  - Morphology (form/structure of abnormality)
  - Finding Site (anatomical location)
- These predictions help evaluate candidate matches

3. **Evaluator**
- The `evaluator` node:
  1. Retrieves full concept details for each candidate
  2. Compares predicted attributes with actual SNOMED relationships
  3. Scores candidates on a 1-5 scale
  4. Provides reasoning for each score
- Candidates scoring ≥4 are added to a shortlist
- A perfect match (score=5) is selected as the final candidate

4. **Refiner**
- If no suitable candidate is found, the `refine` node can modify search terms
- Process continues until either:
  1. A perfect match is found
  2. Maximum revisions are reached

##### Prompts
1. `QUERY_PROMPT` - Generates 1-3 search terms from relation triplet
2. `ATTRIBUTE_PROMPT` - Predicts morphology and finding site for a term
3. `EVALUATION_PROMPT` - Scores candidates on 1-5 scale based on attribute matches

##### State Schema

```python
class AgentState(BaseModel):
    relations: str  # Input relation triplet
    search_terms: List[str]  # Generated search terms
    snomed_candidates: Dict[str, str]  # term -> conceptId mapping
    target_attributes: str  # Predicted morphology & finding site
    evals: List[ScoreCard]  # Evaluation results
    shortlist: List[ScoreCard]  # High-scoring candidates (≥4)
    final_candidate: ScoreCard  # Best matching candidate
    revision_number: int  # Current revision attempt
    max_revisions: int  # Max revision attempts
```

```python
class ScoreCard(BaseModel):
    candidate_term: str
    snomed_id: str
    score: int  # 1-5 rating
    reasoning: str  # Explanation of score
```

##### Usage Considerations
- Requires access to a SNOMED CT terminology server
- Performance depends on server response times
- Quality of matches relies on the underlying LLM's medical knowledge
- Best suited for clinical terms with clear morphology and finding sites