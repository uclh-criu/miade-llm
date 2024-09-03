# miade-llm

MiADE supercharged with LLMs - for the detailed extraction of diagnosis.

This project uses LangServe to deploy langchain chains as REST API endpoints.

## Environment Setup

Currently using [Mixtral-8x7B-instruct-v0.1 hosted by Replicate](https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1/versions) so you need to make sure that `REPLICATE_API_TOKEN` is set in your environment.

Prompts are currently pulled from [LangChain Hub] so you also need to set `LANGCHAIN_API_KEY`.

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
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all endpoints at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Access the playground at `http://127.0.0.1:8000/name-of-package/playground`

Access the endpoints from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/name-of-package")
```
## Chains
### relation-extractor
Extracts relations between concepts found in note and outputs in a JSON-format. Uses MedCAT for NER (requires model).

### RAG chain
placeholder description.