import os
from pydantic import BaseSettings

class RelationExtractorConfig(BaseSettings):
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 100
    temperature: float = 1.0
    replicate_id: str = "mistralai/mixtral-8x7b-instruct-v0.1:5d78bcd7a992c4b793465bcdcf551dc2ab9668d12bb7aa714557a21c1e77041c"
    medcat_model_path: str = os.path.join(os.path.dirname(__file__), "models/miade_mimic_problems_unsupervised_trained_modelpack_w_meta_jun_2023_f25ec9423958e8d6.zip")
    prompt_id: str = "jenniferjiang/extract-medical-entity-relations-base"

    class Config:
        env_prefix = "RELATION_EXTRACTOR_"
        env_file = ".env"
        env_file_encoding = "utf-8"

config = RelationExtractorConfig()