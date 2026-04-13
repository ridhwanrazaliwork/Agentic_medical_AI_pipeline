"""
Agent state definitions and LLM/SLM node configurations
"""

from typing import Annotated, Optional, Dict, List, Any, Tuple
from typing_extensions import TypedDict
from datetime import datetime
import operator
import os
from openai import OpenAI

# Load environment variables - explicit path
from dotenv import load_dotenv

# Get current directory to find .env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(current_dir, ".env")
load_dotenv(env_file)

# HuggingFace transformers for local SLM
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed. Install: pip install transformers torch")


# ===================== STATE SCHEMA =====================

class AgentState(TypedDict):
    """
    Centralized state for the entire agentic workflow
    Tracks data, decisions, and results at each stage
    NOTE: DataFrames stored externally as Parquet files (not in state) to keep state JSON-serializable
    """
    # Task 1: Data Loading
    dataset_path: str  # Path to Parquet file (not DataFrame itself)
    metadata: Dict[str, Any]
    
    # Task 2: EDA
    charts: List[str]  # Base64-encoded PNG images
    correlation_data: Dict[str, Any]
    llm_explanation: str
    chart_metadata: Dict[str, Any]
    
    # Task 3: Cleaning & Classification
    imputation_strategy: str
    strategy_decision_log: str
    cleaned_data_path: str  # Path to Parquet file (not DataFrame itself)
    slm_classifications: List[Dict[str, Any]]
    llm_vs_slm_comparison: Dict[str, Any]
    
    # Workflow tracking
    logs: Annotated[list, operator.add]  # Detailed logs of all decisions
    current_agent: str  # Which agent is currently processing
    errors: List[str]  # Any errors encountered
    timestamp: str  # When workflow started


def create_initial_state() -> AgentState:
    """
    Create fresh initial state for new workflow run
    DataFrames stored as Parquet files in /tmp, paths stored in state
    """
    import tempfile
    temp_dir = tempfile.gettempdir()
    
    return {
        "dataset_path": "",  # Will be populated by load_data_node
        "metadata": {},
        "charts": [],
        "correlation_data": {},
        "llm_explanation": "",
        "chart_metadata": {},
        "imputation_strategy": "",
        "strategy_decision_log": "",
        "cleaned_data_path": "",  # Will be populated by cleaning_strategy_node
        "slm_classifications": [],
        "llm_vs_slm_comparison": {},
        "logs": [f"[{datetime.now().isoformat()}] Workflow initialized"],
        "current_agent": "loader",
        "errors": [],
        "timestamp": datetime.now().isoformat()
    }


# ===================== LLM/SLM CLIENT FACTORIES =====================

def get_azure_client() -> OpenAI:
    """
    Initialize Azure OpenAI client for LLM tasks
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    
    if not endpoint or not api_key:
        raise ValueError(
            "❌ Missing Azure credentials in .env file\n"
            "Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY"
        )
    
    return OpenAI(
        base_url=f"{endpoint}/openai/v1",
        api_key=api_key
    )


def get_local_slm_model() -> Tuple[Any, Any]:
    """
    Initialize local Qwen2.5-0.5B-Instruct model for classification
    Uses HuggingFace transformers library
    
    Returns:
        (model, tokenizer) tuple
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "❌ transformers library not installed\n"
            "Install: pip install transformers torch\n"
            "Or use: conda install -c conda-forge transformers pytorch"
        )
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"📥 Loading {model_name}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✅ Model loaded: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        raise ValueError(f"❌ Failed to load model: {str(e)}\n{model_name}")


def get_deployment_name() -> str:
    """Get Azure deployment name from environment"""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
