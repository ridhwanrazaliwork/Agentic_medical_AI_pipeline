"""
Streamlit UI for Agentic AI Medical Data Analysis Pipeline
Real-time streaming of agent progress + interactive visualization of results
"""

import streamlit as st
import pandas as pd
import base64
from io import BytesIO, StringIO
import os

# Load environment - explicit path
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(current_dir, ".env")
load_dotenv(env_file)
print(f"[DEBUG] Loading .env from: {env_file}")
print(f"[DEBUG] .env exists: {os.path.exists(env_file)}")

from workflow import graph
from agents import create_initial_state

# ===================== PAGE CONFIG =====================

st.set_page_config(
    page_title="AI Medical Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ===================== INITIALIZE SESSION STATE =====================

if "workflow_executed" not in st.session_state:
    st.session_state.workflow_executed = False
    st.session_state.state = None
    st.session_state.logs = []

# ===================== SIDEBAR: AGENT STATUS =====================

st.sidebar.markdown("## 🤖 Agent Status")

# placeholders for real-time updates
status_placeholder = st.sidebar.empty()
log_container = st.sidebar.container()

# ===================== MAIN HEADER =====================

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>🤖 Agentic AI Medical Data Pipeline</h1>
    <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;'>
        Autonomous workflow powered by LangGraph | Task 1-3 with LLM/SLM integration
    </p>
</div>
""", unsafe_allow_html=True)

# ===================== MAIN CONTENT AREA =====================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📊 Workflow Progress")
    progress_placeholder = st.empty()

with col2:
    run_button = st.button("▶️ Run Full Pipeline", key="run_btn", use_container_width=True)

# ===================== RUN WORKFLOW =====================

if run_button:
    st.session_state.workflow_executed = True
    st.session_state.state = None
    st.session_state.logs = []
    
    with st.spinner("🔄 Running agentic workflow..."):
        try:
            # Initialize state
            initial_state = create_initial_state()
            
            # 🔧 FIX: Provide config with thread_id for MemorySaver checkpointing
            config = {"configurable": {"thread_id": "medical_data_pipeline_1"}}
            
            # Run workflow and collect full state at the end
            progress_bar = progress_placeholder.progress(0)
            log_placeholder = st.empty()
            node_count = 5  # load_data, validate, eda, cleaning, report
            node_index = 0
            
            # Use graph.invoke() to get final state directly
            final_state = graph.invoke(initial_state, config=config)
            
            # Save complete state
            st.session_state.state = final_state
            progress_placeholder.success("✅ Workflow complete!")
            
        except Exception as e:
            st.error(f"❌ Workflow failed: {str(e)}")
            import traceback
            st.error(f"**Error Details**: {traceback.format_exc()}")
            st.session_state.logs = [f"Error: {str(e)}"]

# ===================== DISPLAY RESULTS =====================

if st.session_state.workflow_executed and st.session_state.state:
    state = st.session_state.state
    
    # Sidebar logs
    with log_container:
        st.markdown("### 📝 Recent Logs")
        recent_logs = state.get("logs", [])[-5:]
        for log in recent_logs:
            st.write(f"```\n{log}\n```")
        if st.button("View All Logs"):
            st.session_state.show_all_logs = True
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📥 Data", "📈 EDA Charts", "🧹 Cleaning Report", "🤖 LLM vs SLM"]
    )
    
    # ===== TAB 1: DATA =====
    with tab1:
        st.markdown("### Dataset Overview (Task 1)")
        
        if state.get("dataset_path"):
            try:
                df = pd.read_parquet(state["dataset_path"])
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                df = None
        else:
            df = None
        
        if df is not None:
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", state.get("metadata", {}).get("total_rows", "N/A"))
            with col2:
                st.metric("Total Columns", state.get("metadata", {}).get("total_columns", "N/A"))
            with col3:
                st.metric("Missing Values", state.get("metadata", {}).get("missing_values_total", "N/A"))
            with col4:
                st.metric("Missing %", f"{state.get('metadata', {}).get('missing_percent', 'N/A')}%")
            
            st.markdown("#### Data Source")
            st.info(f"✅ Loaded from: {state.get('metadata', {}).get('loaded_from', 'N/A')}")
            st.info(f"✅ Source: {state.get('metadata', {}).get('data_source', 'N/A')}")
            
            st.markdown("#### Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("#### Missing Values by Column")
            missing_by_col = df.isnull().sum().sort_values(ascending=False)
            missing_by_col = missing_by_col[missing_by_col > 0]
            if len(missing_by_col) > 0:
                st.bar_chart(missing_by_col)
            else:
                st.success("✅ No missing values detected!")
        else:
            st.warning("⚠️ No dataset loaded")
    
    # ===== TAB 2: EDA CHARTS =====
    with tab2:
        st.markdown("### Exploratory Data Analysis (Task 2)")
        
        if len(state["charts"]) > 0:
            # Autonomous decision log
            st.markdown("#### Autonomous Decisions")
            col1, col2 = st.columns(2)
            with col1:
                decision = state["chart_metadata"].get("chart_1_decision", "N/A")
                st.info(f"**Chart 1 Decision**: {decision}")
            with col2:
                corr = state["correlation_data"].get("age_sbp_correlation", state["chart_metadata"].get("age_sbp_correlation", "N/A"))
                st.info(f"**Age-SBP Correlation**: {corr}")
            
            # Display charts
            for i, chart_b64 in enumerate(state["charts"], 1):
                st.markdown(f"#### Chart {i}")
                chart_bytes = base64.b64decode(chart_b64)
                st.image(chart_bytes, use_column_width=True)
            
            # LLM explanation
            st.markdown("#### LLM Analysis")
            if state["llm_explanation"]:
                st.markdown(state["llm_explanation"])
            else:
                st.warning("No LLM analysis available")
        else:
            st.warning("⚠️ No charts generated")
    
    # ===== TAB 3: CLEANING REPORT =====
    with tab3:
        st.markdown("### Data Cleaning & Imputation (Task 3)")
        
        # Imputation strategy badge
        if state["imputation_strategy"]:
            st.markdown(f"""
            <div style='background: #d4edda; padding: 1rem; border-radius: 5px; border-left: 4px solid #28a745;'>
                <strong>🎯 Strategy Selected:</strong> <code>{state['imputation_strategy']}</code>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Decision Log**: {state['strategy_decision_log']}")
        
        # Before/after stats
        if state.get("dataset_path") and state.get("cleaned_data_path"):
            try:
                df_before = pd.read_parquet(state["dataset_path"])
                df_after = pd.read_parquet(state["cleaned_data_path"])
                
                st.markdown("#### Missing Values: Before vs After")
                
                missing_before = df_before.isnull().sum().sum()
                missing_after = df_after.isnull().sum().sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Before Imputation", missing_before, delta="")
                with col2:
                    st.metric("After Imputation", missing_after, delta="")
                with col3:
                    reduction = missing_before - missing_after
                    st.metric("Values Filled", reduction, delta=f"{reduction} ↓")
                
                # Download cleaned data
                st.markdown("#### Download Cleaned Dataset")
                csv_buffer = StringIO()
                df_after.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode()
                
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv_bytes,
                    file_name="cleaned_patients_processed.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to load cleaned data: {e}")
        else:
            st.info("⚠️ Cleaned data not available yet")
    
    # ===== TAB 4: LLM vs SLM COMPARISON =====
    with tab4:
        st.markdown("### LLM vs SLM Classification Comparison (Task 3)")
        
        comparison = state.get("llm_vs_slm_comparison", {})
        
        if comparison:
            st.markdown(f"""
            <div style='background: #cce5ff; padding: 1rem; border-radius: 5px; border-left: 4px solid #0066cc;'>
                <strong>📊 Agreement Rate:</strong> <code>{comparison.get('agreement_rate', 'N/A')}%</code> 
                (n={comparison.get('sample_size', 'N/A')} diagnoses)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Analysis**: {comparison.get('comparison_note', 'N/A')}")
            
            # Classification comparison table
            if comparison.get("llm_classifications") and comparison.get("slm_classifications"):
                st.markdown("#### Sample Classifications")
                
                comparison_data = []
                for llm_res, slm_res in zip(
                    comparison["llm_classifications"],
                    comparison["slm_classifications"]
                ):
                    comparison_data.append({
                        "Diagnosis": llm_res.get("diagnosis", "N/A")[:30],
                        "LLM": llm_res.get("classification", "N/A"),
                        "SLM (Qwen)": slm_res.get("classification", "N/A"),
                        "Match": "✅" if llm_res.get("classification") == slm_res.get("classification") else "❌"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # SLM classification breakdown
            if state.get("slm_classifications"):
                st.markdown("#### SLM Classification Distribution")
                slm_classes = [r["classification"] for r in state["slm_classifications"]]
                class_counts = pd.Series(slm_classes).value_counts()
                st.bar_chart(class_counts)
        else:
            st.warning("⚠️ No comparison data available")
    
    # ===== LOGS SIDEBAR =====
    if st.session_state.get("show_all_logs"):
        with st.sidebar:
            st.markdown("### 📋 Complete Logs")
            for log in state.get("logs", []):
                st.code(log)

else:
    if not st.session_state.workflow_executed:
        st.info("👈 Click **Run Full Pipeline** to start the workflow")
    else:
        st.warning("⚠️ Workflow failed or not executed")
