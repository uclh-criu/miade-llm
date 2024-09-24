from langchain.schema.runnable import RunnableSequence
from relation_extractor import chain as relation_extractor_chain


# Combine runnables
chain = RunnableSequence(
    relation_extractor_chain
)
