from vanna.openai import OpenAI_Chat
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv, find_dotenv
import os
import logging
import warnings
import builtins
from typing import Optional
import config
import time
import chromadb.utils.embedding_functions as embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

load_dotenv(find_dotenv())

# Pre-download ONNX model to avoid timeout during training
def pre_download_onnx_model():
    """Pre-download the ONNX model for ChromaDB to prevent timeouts during training"""
    try:
        print("[VANNA DEBUG] Pre-downloading ONNX model for ChromaDB...")
        embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
        embedding_function._download_model_if_not_exists()
        print("[VANNA DEBUG] ✅ ONNX model pre-downloaded successfully")
    except Exception as e:
        print(f"[VANNA DEBUG] ❌ Failed to pre-download ONNX model: {e}")

pre_download_onnx_model()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# Store original print
_original_print = builtins.print

SUPPRESS_PHRASES = [
    "SQL Prompt:",
    "Using model",
    "LLM Response:",
    "Extracted SQL:",
    "tokens (approx)"
]

def filtered_print(*args, **kwargs):
    """Custom print that filters out Vanna's verbose output"""
    text = ' '.join(str(arg) for arg in args)
    if any(phrase in text for phrase in SUPPRESS_PHRASES):
        return
    _original_print(*args, **kwargs)

builtins.print = filtered_print

class VannaModelManager:
    """Manager class to handle Vanna with LLM providers for SQL generation"""

    def __init__(self):
        self.current_provider = self._get_active_provider()
        self.vanna_client = None

    def _get_active_provider(self) -> str:
        """Determine which provider is currently active based on config"""
        if config.USE_VANNA_OPENAI:
            return "openai"
        elif config.USE_VANNA_OLLAMA:
            return "ollama"
        else:
            raise ValueError(
                "No Vanna provider is enabled in config. Set either vanna.openai.enabled or vanna.ollama.enabled to true")

    def get_vanna_class(self, provider: str):
        """Get the appropriate Vanna class based on provider"""
        if provider == "openai":
            class MyVannaOpenAI(ChromaDB_VectorStore, OpenAI_Chat):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    OpenAI_Chat.__init__(self, config=config)

                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def train(self, **kwargs):
                    return super().train(**kwargs)

            return MyVannaOpenAI

        elif provider == "ollama":
            class MyVannaOllama(ChromaDB_VectorStore, Ollama):
                def __init__(self, config=None):
                    ChromaDB_VectorStore.__init__(self, config=config)
                    Ollama.__init__(self, config=config)

                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                def train(self, **kwargs):
                    return super().train(**kwargs)

            return MyVannaOllama

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def initialize_vanna(self, provider: Optional[str] = None):
        """Initialize Vanna with specified provider or use config default"""
        target_provider = provider or self.current_provider

        if target_provider == "openai":
            self._init_openai_vanna()
        elif target_provider == "ollama":
            self._init_ollama_vanna()
        else:
            raise ValueError(f"Unsupported provider: {target_provider}")

        print(f"Vanna initialized with provider: {target_provider}")
        return self.vanna_client

    def _init_openai_vanna(self):
        """Initialize Vanna with OpenAI"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        VannaClass = self.get_vanna_class("openai")
        client_config = {
            'api_key': config.OPENAI_API_KEY,
            'model': config.VANNA_OPENAI_MODEL,
            'allow_llm_to_see_data': config.VANNA_OPENAI_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OPENAI_VERBOSE,
            'http_client': httpx.Client(timeout=60.0)  # Increased timeout
        }

        self.vanna_client = VannaClass(config=client_config)
        self.current_provider = "openai"

    def _init_ollama_vanna(self):
        """Initialize Vanna with Ollama"""
        VannaClass = self.get_vanna_class("ollama")
        self.vanna_client = VannaClass(config={
            'model': config.VANNA_OLLAMA_MODEL,
            'base_url': config.VANNA_OLLAMA_BASE_URL,
            'allow_llm_to_see_data': config.VANNA_OLLAMA_ALLOW_LLM_TO_SEE_DATA,
            'verbose': config.VANNA_OLLAMA_VERBOSE
        })
        self.current_provider = "ollama"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_sql(self, query: str) -> str:
        """Generate SQL from a natural language query"""
        if not self.vanna_client:
            raise ValueError("Vanna client must be initialized before generating SQL")
        try:
            sql = self.vanna_client.generate_sql(query)
            if not sql:
                print("[VANNA DEBUG] Warning: Generated SQL is empty")
            return sql
        except Exception as e:
            print(f"[VANNA DEBUG] Error generating SQL: {e}")
            return ""

    def get_current_provider(self) -> str:
        """Get current active provider"""
        return self.current_provider

    def get_info(self) -> dict:
        """Get current Vanna configuration info"""
        return {
            "provider": self.current_provider,
            "model": config.VANNA_OPENAI_MODEL if self.current_provider == "openai" else config.VANNA_OLLAMA_MODEL,
            "initialized": self.vanna_client is not None
        }


def vanna_train(
        manager,  # Add manager parameter
        ddl: Optional[str] = None,
        documentation: Optional[str] = None,
        question: Optional[str] = None,
        sql: Optional[str] = None
) -> bool:
    """Train Vanna with different types of data"""

    if not manager.vanna_client:
        manager.initialize_vanna()

    def _safe_train(train_func, data, data_type):
        try:
            train_func(data)
            print(f"✅ Trained {data_type} with {manager.current_provider}")
            return True
        except Exception as e:
            print(f"❌ Error training {data_type}: {e}")
            return False

    success = True

    if ddl:
        if not _safe_train(lambda x: manager.vanna_client.train(ddl=x), ddl, "DDL"):
            success = False

    if documentation:
        if not _safe_train(lambda x: manager.vanna_client.train(documentation=x), documentation, "documentation"):
            success = False

    if question and sql:
        if not _safe_train(lambda x: manager.vanna_client.train(question=x[0], sql=x[1]), (question, sql), "SQL pair"):
            success = False

    if not any([ddl, documentation, (question and sql)]):
        raise ValueError("Must provide at least one training data type")

    return success


def generate_sql(manager, query: str) -> str:
    """Generate SQL from a natural language query"""
    return manager.generate_sql(query)


def get_vanna_info(manager) -> dict:
    """Get current Vanna configuration info"""
    return manager.get_info()


# Integrated test code with updated DDLs and documentation
if __name__ == "__main__":
    # Create global manager instance
    vanna_manager = VannaModelManager()
    print("Initializing Vanna SQL Assistant...")

    vn = vanna_manager.initialize_vanna()
    print(f"✅ Vanna initialized with provider: {vanna_manager.current_provider}")

    # Define DDL for [Nodinite].[ods].[Invoice] and [Nodinite].[ods].[Invoice_Line]
    invoice_ddl = """
    CREATE TABLE [Nodinite].[ods].[Invoice] (
        INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
        ISSUE_DATE NVARCHAR(10) NOT NULL,
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
        SUPPLIER_PARTY_NAME NVARCHAR(255),
        SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),
        SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
        SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),
        SUPPLIER_PARTY_CITY NVARCHAR(100),
        SUPPLIER_PARTY_COUNTRY NVARCHAR(2),
        SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
        SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100),
        SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
        SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
        SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
        SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
        CUSTOMER_PARTY_ID NVARCHAR(50),
        CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),
        CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
        CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
        CUSTOMER_PARTY_NAME NVARCHAR(255),
        CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),
        CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),
        CUSTOMER_PARTY_COUNTRY NVARCHAR(2),
        CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
        CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),
        CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
        CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
        CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
        DUE_DATE NVARCHAR(10),
        DOCUMENT_CURRENCY_CODE NVARCHAR(3),
        DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),
        DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
        DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),
        DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),
        DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
        DELIVERY_LOCATION_COUNTRY NVARCHAR(2),
        DELIVERY_PARTY_NAME NVARCHAR(255),
        ACTUAL_DELIVERY_DATE NVARCHAR(10),
        TAX_AMOUNT_CURRENCY NVARCHAR(3),
        TAX_AMOUNT DECIMAL(18,2),
        PERIOD_START_DATE NVARCHAR(10),
        PERIOD_END_DATE NVARCHAR(10),
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,2),
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
        LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,2),
        BUYER_REFERENCE NVARCHAR(100),
        PROJECT_REFERENCE_ID NVARCHAR(100),
        INVOICE_TYPE_CODE NVARCHAR(10),
        NOTE NVARCHAR(MAX),
        TAX_POINT_DATE NVARCHAR(10),
        ACCOUNTING_COST NVARCHAR(100),
        ORDER_REFERENCE_ID NVARCHAR(100),
        ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
        PAYMENT_TERMS_NOTE NVARCHAR(MAX),
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
        BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(10),
        CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),
        DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
        ETL_LOAD_TS NVARCHAR(30)
    );
    """

    invoice_line_ddl = """
    CREATE TABLE [Nodinite].[ods].[Invoice_Line] (
        INVOICE_ID NVARCHAR(50) NOT NULL,
        ISSUE_DATE NVARCHAR(10) NOT NULL,
        SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
        INVOICE_LINE_ID NVARCHAR(50) NOT NULL,
        ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(50),
        ACCOUNTING_COST NVARCHAR(100),
        INVOICED_QUANTITY DECIMAL(18,4),
        INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),
        INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,2),
        INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3),
        INVOICE_PERIOD_START_DATE NVARCHAR(10),
        INVOICE_PERIOD_END_DATE NVARCHAR(10),
        INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),
        INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(10),
        INVOICE_LINE_NOTE NVARCHAR(MAX),
        ITEM_DESCRIPTION NVARCHAR(MAX),
        ITEM_NAME NVARCHAR(255),
        ITEM_TAXCAT_ID NVARCHAR(10),
        ITEM_TAXCAT_PERCENT DECIMAL(5,2),
        ITEM_BUYERS_ID NVARCHAR(100),
        ITEM_SELLERS_ITEM_ID NVARCHAR(100),
        ITEM_STANDARD_ITEM_ID NVARCHAR(100),
        ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),
        ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(50),
        PRICE_AMOUNT DECIMAL(18,2),
        PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),
        PRICE_BASE_QUANTITY DECIMAL(18,4),
        PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),
        PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,2),
        PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),
        ETL_LOAD_TS NVARCHAR(30),
        PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
        FOREIGN KEY (INVOICE_ID) REFERENCES [Nodinite].[ods].[Invoice](INVOICE_ID)
    );
    """

    # Define documentation with sample data aligned with schema and context
    invoice_doc = """
    Sample data from [Nodinite].[ods].[Invoice] table (first 5 rows):
    INVOICE_ID,ISSUE_DATE,SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID,SUPPLIER_PARTY_NAME,CUSTOMER_PARTY_NAME,DOCUMENT_CURRENCY_CODE,LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
    INV001,2025-01-01,COMP001,JA Hotel Karlskrona,Region Västerbotten,SEK,12500.00
    INV002,2025-01-02,COMP001,JA Hotel Karlskrona,Stockholms Stad,SEK,9800.50
    INV003,2025-01-03,COMP002,Visma Draftit AB,Region Skåne,SEK,45000.75
    INV004,2025-01-04,COMP003,Abbott Scandinavia,Västra Götaland,SEK,32000.00
    INV005,2025-01-05,COMP004,Nordic IT Solutions AB,Region Västerbotten,SEK,15000.25
    """

    invoice_line_doc = """
    Sample data from [Nodinite].[ods].[Invoice_Line] table (first 5 rows):
    INVOICE_ID,ISSUE_DATE,SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID,INVOICE_LINE_ID,ITEM_NAME,INVOICED_QUANTITY,PRICE_AMOUNT,INVOICED_LINE_EXTENSION_AMOUNT,ITEM_TAXCAT_PERCENT
    INV001,2025-01-01,COMP001,LINE001,Hotel Accommodation,2.0000,5000.00,10000.00,25.00
    INV001,2025-01-01,COMP001,LINE002,Training Services,1.0000,2500.00,2500.00,25.00
    INV002,2025-01-02,COMP001,LINE003,Hotel Accommodation,1.0000,9800.50,9800.50,25.00
    INV003,2025-01-03,COMP002,LINE004,Software License,1.0000,45000.75,45000.75,25.00
    INV004,2025-01-04,COMP003,LINE005,Medical Supplies,10.0000,3200.00,32000.00,25.00
    """

    # Train Vanna with DDL and documentation
    print("\n📚 Training Vanna with DDL and documentation...")
    print("Training with Invoice DDL...")
    if vanna_train(vanna_manager, ddl=invoice_ddl):
        print("✅ Successfully trained Invoice DDL")
    else:
        print("❌ Failed to train Invoice DDL")

    print("Training with Invoice_Line DDL...")
    if vanna_train(vanna_manager, ddl=invoice_line_ddl):
        print("✅ Successfully trained Invoice_Line DDL")
    else:
        print("❌ Failed to train Invoice_Line DDL")

    print("Training with Invoice documentation (first 5 rows)...")
    if vanna_train(vanna_manager, documentation=invoice_doc):
        print("✅ Successfully trained Invoice documentation")
    else:
        print("❌ Failed to train Invoice documentation")

    print("Training with Invoice_Line documentation (first 5 rows)...")
    if vanna_train(vanna_manager, documentation=invoice_line_doc):
        print("✅ Successfully trained Invoice_Line documentation")
    else:
        print("❌ Failed to train Invoice_Line documentation")

    # Auto-train on startup if enabled
    if config.VANNA_AUTO_TRAIN or config.VANNA_TRAIN_ON_STARTUP:
        print(
            "\n🔄 Auto-training enabled. Additional training can be done manually with vanna_train(vanna_manager, ...).")
    else:
        print("\n🔄 Auto-training is disabled. Use vanna_train(vanna_manager, ...) to manually train the model.")

    # Restore original print
    builtins.print = _original_print

    print(f"\n🤖 Vanna SQL Assistant initialized with {vanna_manager.current_provider} provider")
    print(f"📊 Current model: {get_vanna_info(vanna_manager)['model']}")

    # Generate SQL for the query
    query = "List the first two items on the invoice table"
    print(f"\n📝 Generating SQL for query: '{query}'")
    sql = generate_sql(vanna_manager, query)
    print("✅ Generated SQL:")
    print(sql)

    print("\n💡 Available commands:")
    print("  • generate_sql('Natural language query') to generate SQL")
    print("  • vanna_train() to train with DDL, documentation, or question-SQL pairs")
    print("  • get_vanna_info() to see current configuration")
    print("  • Type 'exit' to quit")