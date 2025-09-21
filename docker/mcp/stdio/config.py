import os
from dotenv import load_dotenv
import json
from typing import Any, Dict, List
from langchain.chat_models.base import init_chat_model
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI



global_config_path = '/.omnigate/config.json'
script_dir = os.path.dirname(os.path.abspath(__file__))
local_config_path = os.path.join(script_dir, 'config.json')

load_dotenv()


def get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely traverse a nested dict using a list of keys.
    Returns default if any key is missing.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_bool(value: Any, default: bool = False) -> bool:
    """
    Coerce value to bool safely. Accepts Python bool, truthy strings, and ints.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return default


def to_int(value: Any, default: int) -> int:
    """
    Coerce value to int safely; return default on failure.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Load the JSON file
config = None
if os.path.exists(global_config_path):
    with open(global_config_path, "r") as file:
        config = json.load(file)

    CONFIG_FILE_SOURCE = 'global'

elif os.path.exists(local_config_path):
    with open(local_config_path, "r") as file:
        config = json.load(file)

    CONFIG_FILE_SOURCE = 'local'

else:
    assert False, f"No config file found at {global_config_path} or {local_config_path}"

# Access the app version
APP_VERSION = config["app"]["version"]
print(f"App version: {APP_VERSION}")

# --- 3) Read top-level app info safely ---
APP_NAME = get_nested(config, ["app", "name"], "app")
APP_VERSION = get_nested(config, ["app", "version"], "0.0.0")
APP_DESCRIPTION = get_nested(config, ["app", "description"], "")
RUN_LOCAL = to_bool(get_nested(config, ["app", "run_local"], False), default=False)
DEBUG = to_bool(get_nested(config, ["app", "debug"], False), default=False)

MCP_DATABASE_URL = get_nested(config, ["mcp", "database_url"], None)
# --- 4) Environment settings (mode, debug) ---

# --- 5) Ollama settings (enabled, base_url, port, model) ---
USE_OLLAMA = to_bool(get_nested(config, ["ollama", "enabled"], False), default=False)
OLLAMA_HOST = get_nested(config, ["ollama", "host"], "http://localhost")
OLLAMA_PORT = to_int(get_nested(config, ["ollama", "port"], 11434), default=11434)
OLLAMA_BASE_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}"

# --- 6) OpenAI settings ---
USE_OPENAI = to_bool(get_nested(config, ["openai", "enabled"], False), default=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if USE_OLLAMA:
    AGENT_MODEL_NAME = get_nested(config, ["ollama", "agent_model_name"], "qwen3:latest")
    EMBEDDINGS_MODEL_NAME = get_nested(config, ["ollama", "embeddings_model_name"],
                                       "jeffh/intfloat-multilingual-e5-large:q8_0")

if USE_OPENAI:
    AGENT_MODEL_NAME = get_nested(config, ["openai", "agent_model_name"], "gpt-4o-mini")
    EMBEDDINGS_MODEL_NAME = get_nested(config, ["openai", "embeddings_model_name"], "text-embedding-3-large")

USE_MISTRAL = to_bool(get_nested(config, ["mistral", "enabled"], False), default=False)

if USE_MISTRAL:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL_NAME = get_nested(config, ["mistral", "model_name"], "devstral-medium-2507")
    MISTRAL_BASE_URL = get_nested(config, ["mistral", "base_url"], "https://api.mistral.ai/v1")
    MISTRAL_CHAT_COMPLETION_ENDPOINT = get_nested(config, ["mistral", "chat_completion_endpoint"], "/chat/completions")
    EMBEDDINGS_MODEL_NAME = get_nested(config, ["mistral", "embeddings_model_name"], "text-embedding-3-large")

# --- 7) Qdrant settings ---
QDRANT_HOST = get_nested(config, ["qdrant", "host"], "localhost")
QDRANT_PORT = to_int(get_nested(config, ["qdrant", "port"], 6333), default=6333)
QDRANT_RESULT_LIMIT = to_int(get_nested(config, ["qdrant", "result_limit"], 2), default=2)

COSMIC_DATABASE_COLLECTION = get_nested(config, ["cosmic_database", "collection_name"], "cosmic_documents")
COSMIC_DATABASE_DOCUMENT_PATH = get_nested(config, ["cosmic_database", "document_path"], "cosmic_documents")

COSMIC_DATABASE_COLLECTION_NAME = f"{COSMIC_DATABASE_COLLECTION}-{EMBEDDINGS_MODEL_NAME}"
print(f"Using cosmic database collection name: {COSMIC_DATABASE_COLLECTION_NAME}")
COSMIC_DATABASE_COLLECTION_NAME = COSMIC_DATABASE_COLLECTION_NAME.replace(':', '-').replace('/', '-')

# --- 8) Grafana logging settings ---
GRAFANA_ENABLED = to_bool(get_nested(config, ["grafana", "enabled"], False), default=False)
GRAFANA_LEVEL = get_nested(config, ["grafana", "level"], "info")
GRAFANA_LOG_TO_FILE = to_bool(get_nested(config, ["grafana", "log_to_file"], False), default=False)

# --- 9) Feature flags (default to False if missing) ---
FEATURE_COSMIC_AGENT = to_bool(get_nested(config, ["features", "cosmic_agent"], False), default=False)
FEATURE_CREATE_TICKET = to_bool(get_nested(config, ["features", "create_ticket"], False), default=False)
FEATURE_WHO_AM_I = to_bool(get_nested(config, ["features", "who_am_i"], False), default=False)

# --- 10) LangSmith settings ---
LANGSMITH_ENABLED = to_bool(get_nested(config, ["langsmith", "enabled"], False), default=False)
LANGSMITH_TRACING = to_bool(get_nested(config, ["langsmith", "tracing"], False), default=False)

# 11) Database configuration
MCP_DATABASE_PATH = get_nested(config, ["mcp", "database_path"], "social_media_database.db")
MCP_DOCKER_DATABASE_PATH = get_nested(config, ["mcp", "docker_database_path"],
                                      "/app/database_data/social_media_database.db")
MCP_DATABASE_SERVER_URL = get_nested(config, ["mcp", "database_server_url"], "http://localhost:8762")
MCP_DOCKER_DATABASE_SERVER_URL = get_nested(config, ["mcp", "docker_database_server_url"],
                                            "http://database_server:8762")
MCP_PROVIDER_OPENAI = to_bool(get_nested(config, ["mcp", "provider", "openai"], False), default=False)
MCP_PROVIDER_OLLAMA = to_bool(get_nested(config, ["mcp", "provider", "ollama"], False), default=False)
MCP_PROVIDER_MISTRAL = to_bool(get_nested(config, ["mcp", "provider", "mistral"], False), default=False)

if MCP_PROVIDER_OLLAMA:
    MCP_AGENT_MODEL_NAME = get_nested(config, ["ollama", "agent_model_name"], "qwen3:latest")
if MCP_PROVIDER_OPENAI:
    MCP_AGENT_MODEL_NAME = get_nested(config, ["openai", "agent_model_name"], "gpt-4o-mini")

if MCP_PROVIDER_MISTRAL:
    MCP_AGENT_MODEL_NAME = get_nested(config, ["mistral", "model_name"], "devstral-medium-2507")

# --- 12) Example usage (remove or adapt in production) ---
if USE_OLLAMA:
    AGENT_MODEL = init_chat_model(
        AGENT_MODEL_NAME,
        model_provider="ollama",
        base_url=OLLAMA_BASE_URL,
        reasoning=False,
        model_kwargs={
            "stream": True,
            "tool_choice": "auto",
        }
    )

    KEYWORD_AGENT = ChatOllama(
        model=MCP_AGENT_MODEL_NAME,
        temperature=0,
        stream=True,
        base_url=OLLAMA_BASE_URL
    )

elif USE_OPENAI:
    # AGENT_MODEL=AGENT_MODEL_NAME
    AGENT_MODEL = init_chat_model(
        AGENT_MODEL_NAME,
        model_kwargs={
            "stream": True,
            "tool_choice": "auto",
        }
    )

    KEYWORD_AGENT = ChatOpenAI(
        model=MCP_AGENT_MODEL_NAME,
        temperature=0,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY")
    )

elif USE_MISTRAL:
    AGENT_MODEL = init_chat_model(
        MISTRAL_MODEL_NAME,
        model_provider="mistralai",
        base_url=MISTRAL_BASE_URL,
        api_key=MISTRAL_API_KEY,
        temperature=0.1,
        model_kwargs={
            "stream": True,
            "tool_choice": "auto",
        }
    )

    KEYWORD_AGENT = init_chat_model(
        MCP_AGENT_MODEL_NAME,
        model_provider="mistralai",
        base_url=MISTRAL_BASE_URL,
        api_key=MISTRAL_API_KEY,
        temperature=0.1
    )

    print(f"Got Model: {AGENT_MODEL}")

else:
    assert False, "USE_OLLAMA or USE_OPENAI must be enabled in the config"

#  Jira configuration
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_PROJECT_KEY = get_nested(config, ["jira", "project_key"], "HEAL")
JIRA_ISSUE_TYPE = get_nested(config, ["jira", "issue_type"], "Task")
CREATE_ISSUE_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue"
TRANSITION_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{{}}/transitions"

# Headers for API requests
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def summarize():
    print(f"App name: {APP_NAME}")
    print(f"App version: {APP_VERSION}")
    print(f"use openai: {USE_OPENAI}")
    print(f"use ollama: {USE_OLLAMA}")
    print(f"debug: {DEBUG}")
    print(
        f"using {CONFIG_FILE_SOURCE} config file at: {global_config_path if CONFIG_FILE_SOURCE == 'global' else local_config_path}")

    print(f"using agent model: {AGENT_MODEL_NAME}")
    print(f"using embeddings model: {EMBEDDINGS_MODEL_NAME}")
    if USE_OLLAMA:
        print(f"using OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

    print(f"using cosmic database collection: {COSMIC_DATABASE_COLLECTION_NAME}")

# summarize()
