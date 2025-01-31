from pydantic import BaseSettings
from relation_extractor.config import RelationExtractorConfig


class AppConfig(BaseSettings):
    relation_extractor: RelationExtractorConfig = RelationExtractorConfig()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = AppConfig()
