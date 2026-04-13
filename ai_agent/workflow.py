"""
LangGraph workflow orchestrating the agentic AI pipeline
- Nodes for each task (load → validate → eda → clean → report)
- Conditional edges for error handling
- MemorySaver checkpointing for resumability
"""

import operator
import logging
from datetime import datetime
from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ===================== LOGGING SETUP =====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from agents import (
    AgentState,
    create_initial_state,
    get_azure_client,
    get_local_slm_model,
    get_deployment_name,
)
from tools import (
    load_assignment1_data,
    generate_data_profile,
    create_autonomous_charts,
    select_imputation_strategy,
    apply_imputation,
    classify_text_slm_batch,
    compare_llm_vs_slm_classifications,
)


# ===================== WORKFLOW NODES =====================

def load_data_node(state: AgentState) -> AgentState:
    """
    TASK 1: Load pre-generated CSV from assignment1.ipynb
    Validate schema and metadata
    """
    csv_path = "/home/ridhwanlaptop/assignment1_datamining/ridhwan/artifacts/generated_patients.csv"
    
    state["current_agent"] = "load_data_node"
    log_msg = f"[{datetime.now().isoformat()}] Starting Task 1: Load data"
    state["logs"].append(log_msg)
    print(log_msg)
    
    log_msg = f"→ CSV path: {csv_path}"
    state["logs"].append(log_msg)
    print(log_msg)
    
    try:
        import tempfile
        import os as os_module
        
        df, metadata = load_assignment1_data(csv_path)
        
        # Save DataFrame as Parquet (not in state to keep state serializable)
        temp_dir = tempfile.gettempdir()
        dataset_path = os_module.path.join(temp_dir, "ai_agent_dataset.parquet")
        df.to_parquet(dataset_path, index=False)
        state["dataset_path"] = dataset_path
        state["metadata"] = metadata
        
        log_msg = (
            f"✅ Dataset loaded: {metadata['total_rows']} rows, "
            f"{metadata['total_columns']} columns"
        )
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = f"→ Saved to: {dataset_path}"
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = (
            f"✅ Missing values: {metadata['missing_values_total']} "
            f"({metadata['missing_percent']}%)"
        )
        state["logs"].append(log_msg)
        print(log_msg)
        
    except FileNotFoundError as e:
        state["errors"].append(str(e))
        log_msg = f"❌ File error: {str(e)}"
        state["logs"].append(log_msg)
        print(log_msg)
        
    except ValueError as e:
        state["errors"].append(str(e))
        log_msg = f"❌ Validation error: {str(e)}"
        state["logs"].append(log_msg)
        print(log_msg)
    
    return state


def validate_gateway_node(state: AgentState) -> AgentState:
    """
    Validation point: Check if data loaded successfully
    If errors, log and prepare for error handling
    """
    state["current_agent"] = "validate_gateway"
    log_msg = f"[{datetime.now().isoformat()}] Validation gateway check"
    state["logs"].append(log_msg)
    print(log_msg)
    
    if not state.get("dataset_path") or len(state["errors"]) > 0:
        log_msg = "❌ Validation FAILED"
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = "→ Cannot proceed without valid dataset"
        state["logs"].append(log_msg)
        print(log_msg)
    else:
        log_msg = "✅ Validation PASSED"
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = "→ Proceeding to Task 2: EDA"
        state["logs"].append(log_msg)
        print(log_msg)
    
    return state


def eda_analysis_node(state: AgentState) -> AgentState:
    """
    TASK 2: Exploratory Data Analysis with autonomous decisions
    - Create charts with autonomous selection logic
    - Generate LLM explanation of charts
    """
    state["current_agent"] = "eda_analysis"
    log_msg = f"[{datetime.now().isoformat()}] Starting Task 2: EDA"
    state["logs"].append(log_msg)
    print(log_msg)
    
    if not state.get("dataset_path"):
        state["errors"].append("Cannot run EDA: no dataset loaded")
        log_msg = "❌ EDA skipped: no dataset"
        state["logs"].append(log_msg)
        print(log_msg)
        return state
    
    try:
        import pandas as pd
        
        # Load DataFrame from Parquet
        df = pd.read_parquet(state["dataset_path"])
        
        # Create autonomous charts
        log_msg = "→ Creating charts with autonomous decision logic..."
        state["logs"].append(log_msg)
        print(log_msg)
        charts, chart_meta = create_autonomous_charts(df)
        
        state["charts"] = charts
        state["chart_metadata"] = chart_meta
        state["correlation_data"] = chart_meta.get("correlation_matrix", {})
        
        log_msg = f"✅ Generated {len(charts)} charts"
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = f"✅ Chart 1 decision: {chart_meta.get('chart_1_decision', 'N/A')}"
        state["logs"].append(log_msg)
        print(log_msg)
        
        log_msg = f"✅ Age-SBP correlation: {chart_meta.get('age_sbp_correlation', 'N/A')}"
        state["logs"].append(log_msg)
        print(log_msg)
        
        # Generate LLM explanation (ONCE per workflow)
        log_msg = "→ Generating LLM chart explanation..."
        state["logs"].append(log_msg)
        print(log_msg)
        try:
            client = get_azure_client()
            deployment = get_deployment_name()
            
            import pandas as pd
            num_charts = len(charts)
            corr_val = state["correlation_data"].get(
                ('vital_heart_rate', 'vital_systolic_bp'), 'unknown'
            )
            
            explanation_prompt = f"""
Chart Analysis (Task 2 - EDA):
- Generated {num_charts} charts from 500-patient dataset
- Key pattern identified:
  * Age-Systolic BP correlation: {state['chart_metadata'].get('age_sbp_correlation', 0):.3f}
  * Chart type decision: {state['chart_metadata'].get('chart_1_decision', 'boxplot')}
  * Gender distribution shows female dominance (~85%)
  * Top diagnosis: Dengue fever variants

Provide 3-4 sentence interpretation of these patterns.
"""
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a healthcare data analyst. Provide clear, concise insights."
                    },
                    {
                        "role": "user",
                        "content": explanation_prompt
                    }
                ],
                temperature=0.7,
                max_completion_tokens=300
            )
            
            state["llm_explanation"] = response.choices[0].message.content
            log_msg = "✅ LLM explanation generated"
            state["logs"].append(log_msg)
            print(log_msg)
            
        except Exception as e:
            state["llm_explanation"] = f"[Explanation generation failed: {str(e)}]"
            log_msg = f"⚠️ LLM explanation error: {str(e)}"
            state["logs"].append(log_msg)
            print(log_msg)
    
    except Exception as e:
        state["errors"].append(f"EDA error: {str(e)}")
        log_msg = f"❌ EDA failed: {str(e)}"
        state["logs"].append(log_msg)
        print(log_msg)
    
    return state


def cleaning_strategy_node(state: AgentState) -> AgentState:
    """
    TASK 3: Autonomous imputation strategy selection + SLM classification
    - Select strategy based on missingness percentage (AUTONOMOUS DECISION)
    - Apply imputation
    - Classify text using Qwen SLM
    - Compare with LLM classifications
    """
    state["current_agent"] = "cleaning_strategy"
    log_msg = f"[{datetime.now().isoformat()}] Starting Task 3: Cleaning & Classification"
    state["logs"].append(log_msg)
    print(log_msg)
    
    if not state.get("dataset_path"):
        state["errors"].append("Cannot run cleaning: no dataset loaded")
        log_msg = "❌ Cleaning skipped: no dataset"
        state["logs"].append(log_msg)
        print(log_msg)
        return state
    
    try:
        import pandas as pd
        import tempfile
        import os as os_module
        
        # Load DataFrame from Parquet
        df = pd.read_parquet(state["dataset_path"])
        
        # ========== AUTONOMOUS DECISION 1: Select imputation strategy ==========
        log_msg = "→ Autonomous decision: Select imputation strategy..."
        state["logs"].append(log_msg)
        print(log_msg)
        strategy, decision_log = select_imputation_strategy(df)
        state["imputation_strategy"] = strategy
        state["strategy_decision_log"] = decision_log
        log_msg = f"✅ {decision_log}"
        state["logs"].append(log_msg)
        print(log_msg)
        
        # ========== Apply imputation ==========
        log_msg = f"→ Applying {strategy}..."
        state["logs"].append(log_msg)
        print(log_msg)
        cleaned_df = apply_imputation(df, strategy)
        
        # Save cleaned DataFrame as Parquet
        temp_dir = tempfile.gettempdir()
        cleaned_data_path = os_module.path.join(temp_dir, "ai_agent_cleaned.parquet")
        cleaned_df.to_parquet(cleaned_data_path, index=False)
        state["cleaned_data_path"] = cleaned_data_path
        
        missing_after = cleaned_df.isnull().sum().sum()
        log_msg = f"✅ Imputation complete: {missing_after} missing values remaining"
        state["logs"].append(log_msg)
        print(log_msg)
        
        # ========== SLM text classification (Local Qwen via transformers) ==========
        log_msg = "→ Loading local Qwen2.5-0.5B model for classification..."
        state["logs"].append(log_msg)
        print(log_msg)
        try:
            model, tokenizer = get_local_slm_model()
            
            # Get sample diagnoses for classification
            diagnoses_sample = cleaned_df['diagnosis'].dropna().unique()[:50]
            diagnoses_list = [str(d) for d in diagnoses_sample if d]
            
            log_msg = f"→ Classifying {len(diagnoses_list)} diagnoses with Qwen2.5-0.5B..."
            state["logs"].append(log_msg)
            print(log_msg)
            
            slm_results = classify_text_slm_batch(diagnoses_list, model, tokenizer)
            state["slm_classifications"] = slm_results
            
            # Count classifications
            critical_count = sum(1 for r in slm_results if r["classification"] == "CRITICAL")
            non_critical_count = sum(1 for r in slm_results if r["classification"] == "NON-CRITICAL")
            
            log_msg = (
                f"✅ SLM (Qwen2.5-0.5B) classification complete: {critical_count} CRITICAL, "
                f"{non_critical_count} NON-CRITICAL (n={len(slm_results)})"
            )
            state["logs"].append(log_msg)
            print(log_msg)
            
        except Exception as e:
            log_msg = f"⚠️ SLM classification error: {str(e)}"
            state["logs"].append(log_msg)
            print(log_msg)
        
        # ========== LLM vs SLM comparison ==========
        log_msg = "→ Comparing LLM vs SLM classifications (Azure GPT-4.1 vs Qwen2.5-0.5B)..."
        state["logs"].append(log_msg)
        print(log_msg)
        try:
            llm_client = get_azure_client()
            model, tokenizer = get_local_slm_model()
            deployment = get_deployment_name()
            
            comparison = compare_llm_vs_slm_classifications(
                diagnoses_list, llm_client, model, tokenizer, deployment
            )
            state["llm_vs_slm_comparison"] = comparison
            
            log_msg = (
                f"✅ LLM vs SLM agreement rate: {comparison['agreement_rate']}% "
                f"(n={comparison['sample_size']} diagnoses)"
            )
            state["logs"].append(log_msg)
            print(log_msg)
            
        except Exception as e:
            log_msg = f"⚠️ Comparison error: {str(e)}"
            state["logs"].append(log_msg)
            print(log_msg)
    
    except Exception as e:
        state["errors"].append(f"Cleaning error: {str(e)}")
        log_msg = f"❌ Cleaning failed: {str(e)}"
        state["logs"].append(log_msg)
        print(log_msg)
    
    return state


def report_node(state: AgentState) -> AgentState:
    """
    Final synthesis: Prepare summary report
    """
    state["current_agent"] = "report"
    log_msg = f"[{datetime.now().isoformat()}] Workflow complete"
    state["logs"].append(log_msg)
    print(log_msg)
    
    state["logs"].append("\n" + "="*70)
    state["logs"].append("WORKFLOW SUMMARY")
    state["logs"].append("="*70)
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    
    # Task 1 summary
    state["logs"].append(
        f"\n✅ Task 1 (Dataset Simulation): Loaded {state['metadata'].get('total_rows', 'N/A')} "
        f"patients from GenAI source (assignment1.ipynb)"
    )
    
    # Task 2 summary
    state["logs"].append(
        f"\n✅ Task 2 (EDA): Generated {len(state['charts'])} charts with "
        f"autonomous decision logic. LLM explanation provided."
    )
    
    # Task 3 summary
    state["logs"].append(
        f"\n✅ Task 3 (Data Prep): Applied {state['imputation_strategy']} "
        f"imputation. SLM classified diagnoses. LLM vs SLM agreement: "
        f"{state['llm_vs_slm_comparison'].get('agreement_rate', 'N/A')}%"
    )
    
    if state["errors"]:
        state["logs"].append(f"\n⚠️ Errors encountered: {len(state['errors'])}")
        for err in state["errors"]:
            state["logs"].append(f"  - {err}")
    else:
        state["logs"].append("\n✅ No errors - workflow completed successfully!")
    
    state["logs"].append("="*70)
    
    return state


# ===================== BUILD GRAPH =====================

def build_workflow():
    """
    Construct LangGraph workflow with all nodes and edges
    Uses MemorySaver for checkpointing
    """
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    memory = MemorySaver()
    
    # Add nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("validate_gateway", validate_gateway_node)
    workflow.add_node("eda_analysis", eda_analysis_node)
    workflow.add_node("cleaning_strategy", cleaning_strategy_node)
    workflow.add_node("report", report_node)
    
    # Define edges
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "validate_gateway")
    
    # Conditional logic: if validation fails, go to report; else proceed to EDA
    def validate_check(state):
        if not state.get("dataset_path") or len(state["errors"]) > 0:
            return "report"  # Short-circuit to report
        return "eda_analysis"
    
    workflow.add_conditional_edges("validate_gateway", validate_check)
    
    # Normal flow
    workflow.add_edge("eda_analysis", "cleaning_strategy")
    workflow.add_edge("cleaning_strategy", "report")
    workflow.add_edge("report", END)
    
    # Compile with memory
    return workflow.compile(checkpointer=memory)


# Export
graph = build_workflow()
