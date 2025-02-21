from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from relation_extractor import chain as relation_extrator_chain
from coder import chain as coder_chain

from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


class Input(BaseModel):
    note: str


# Edit this to add the chain you want to add
add_routes(
    app,
    relation_extrator_chain.with_types(input_type=Input),
    path="/relation_extractor_chain",
)
add_routes(
    app,
    coder_chain.with_types(input_type=Input),
    path="/coder_chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
