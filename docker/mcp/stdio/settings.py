"""
    This file handles the settings for the different components.
"""
import os
from typing import Tuple, Type, Optional
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource
)

class MSSQLCredentials(BaseModel):
    """
        MS-SQL credential model. username & password 
        optional when using Windows Auth.
    """
    database: str
    host: str
    instance: Optional[str] = ''
    username: Optional[str] = None
    password: Optional[str] = None

class PostgresCredentials(BaseModel):
    """
        Postgres credential model.
    """
    pg_database: str
    pg_host: str
    pg_port: int = 5432
    pg_username: str
    pg_password: str


class OllamaSettings(BaseModel):
    """
        Ollama settings model.
    """
    llm_model: str = Field(default="mistral:7b", description="Ollama model name")
    embedding_model: str = Field(default="embeddinggemma:300m", description="Ollama embedding model name")
    base_url: str = Field(default="http://localhost:11434/", description="Ollama server base URL")
    direct_url: Optional[str] = Field(default=None, description="Direct URL to the ollama instance if needed")


class SystemSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',                                  
        env_file=os.path.join(os.path.dirname(__file__), '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )

    #mssql: MSSQLCredentials
    postgres: Optional[PostgresCredentials] = None
    ollama: Optional[OllamaSettings] = None

    path_to_pdf: Optional[str] = None
    #pdf_load_limit: int = Field(default=100, description="Maximum number of PDF files to load")
    #chunk_size: int = Field(default=1000, description="Chunk size for text splitting")
    #chunk_overlap: int = Field(default=200, description="Chunk overlap for text splitting")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings
        )


if __name__ == "__main__":
    settings = SystemSettings()
    print(settings)

