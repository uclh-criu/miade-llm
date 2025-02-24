from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from relation_extractor import chain as relation_extrator_chain
from coder import chains as coder_chains
from coder import models

from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/models")
async def model_list():
    return {
        "model_endpoints": [
            {
                "name": f"{provider}: {name}",
                "endpoint":f"{provider}-{name}/invoke"
            }

            for name, provider in models
        ]
    }

class Input(BaseModel):
    note: str


# Edit this to add the chain you want to add
add_routes(
    app,
    relation_extrator_chain.with_types(input_type=Input),
    path="/relation_extractor_chain",
)
for name, chain in coder_chains:
    add_routes(
        app,
        chain.with_types(input_type=Input),
        path=f"/{name}",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
