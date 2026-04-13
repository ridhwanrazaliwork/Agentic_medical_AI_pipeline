# Agentic AI Medical Data Pipeline – Assignment 1 Extension

## 📋 Overview

This application demonstrates autonomous AI agent workflows using **LangGraph** and **Streamlit**. It processes the medical patient dataset from **WQD7005 Assignment 1** and showcases how an agent can make intelligent, context-aware decisions throughout a multi-step data pipeline.

**Key Innovation**: Rather than hardcoding every step, the agent autonomously decides:
- Which charts to generate (based on data correlation)
- Which imputation strategy to use (based on missing data percentage)
- How LLM and SLM models compare in classification tasks

---

## 🏗️ Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Loaded Patient CSV from assignment1 (500 records)               │
│ └── patient_id, age, gender, vitals, diagnosis, symptoms       │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │ LOAD DATA NODE            │
        │ ✓ Validate schema         │
        │ ✓ Check for 500 records   │
        │ ✓ Generate data profile   │
        └────────────────────┬──────┘
                             │
        ┌────────────────────▼──────────────┐
        │ VALIDATE GATEWAY                  │
        │ ✓ If loaded OK → EDA              │
        │ ✓ If error → Fail                 │
        └────────────────────┬──────────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │ EDA ANALYSIS NODE                     │
        │ ✓ Autonomous chart selection:         │
        │   - If |correlation(age,BP)| > 0.6   │
        │     → Scatter plot + trendline       │
        │   - Else → Boxplot                   │
        │ ✓ Always: Heatmap + diagnosis dist.  │
        │ ✓ LLM generates trend insights       │
        └────────────────────┬──────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ CLEANING STRATEGY NODE                      │
        │ ✓ Count missing data %                      │
        │ ✓ Autonomous strategy selection:            │
        │   - <5% missing → Simple mean imputation    │
        │   - 5-20% → KNN (K=5, nan_euclidean)        │
        │   - >20% → MICE (IterativeImputer)          │
        │ ✓ Apply imputation                          │
        │ ✓ SLM classifies diagnoses (Qwen API)       │
        │ ✓ LLM vs SLM comparison (20 samples)        │
        └────────────────────┬──────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │ REPORT NODE                                 │
        │ ✓ Aggregate all results (before/after)      │
        │ ✓ Log completion metrics                    │
        │ ✓ Package cleaned DataFrame                 │
        └─────────────────────────────────────────────┘
```

### Autonomous Decisions

#### Decision 1: Chart Type Selection (Task 2 – EDA)

**Input**: Correlation between Age and Blood Pressure  
**Logic**:
```python
if abs(correlation) > 0.6:
    → Scatter plot with trendline (strong linear relationship)
else:
    → Boxplot by age groups (weak/no linear relationship)
```
**Always**: Heatmap of all numerical columns + diagnosis distribution chart

**Rationale**: Correlation threshold of 0.6 indicates moderate-strong linear relationship worth visualizing as scatter; weaker relationships better shown via distribution.

#### Decision 2: Imputation Strategy Selection (Task 3 – Data Cleaning)

**Input**: Percentage of missing values in dataset  
**Logic**:
```python
if missing_percent < 5:
    strategy = "Simple Mean Imputation"
    → Fast, minimal bias when missingness is MCAR
elif 5 <= missing_percent <= 20:
    strategy = "KNN Imputation (K=5)"
    → Preserves local structure, moderate complexity
else:  # > 20%
    strategy = "MICE (IterativeImputer)"
    → Iterative multiple imputation, handles MAR/MNAR
```

**Rationale**: 
- <5%: Missingness negligible; simple methods suffice
- 5-20%: KNN balances accuracy & speed
- >20%: MICE accounts for complex patterns

### Model Orchestration

**LLM**: Azure OpenAI gpt-4.1 deployment
- **Task 1**: Generate natural-language profile of dataset (1 call per workflow)
- **Task 2**: Analyze EDA charts & provide trend insights (1 call)
- **Task 3**: Compare LLM classification with SLM on 20 sample diagnoses (1 call)
- **Total LLM Calls**: ~3/workflow (budget-conscious)

**SLM**: Qwen2.5-1.5B-Instruct (HuggingFace Router API)
- **Task 3**: Classify diagnoses as "CRITICAL" vs "NON-CRITICAL" on full dataset (~500 samples)
- **Rationale**: Lightweight, cost-efficient for repetitive classification tasks

**Comparison**: LLM (gpt-4.1) vs SLM (Qwen2.5) on 20 sample diagnoses
- Both classify same samples
- Compute agreement % (how often they agree)
- Display side-by-side results in UI

---

## 📁 Project Structure

```
ai_agent/
├── requirements.txt          # Python dependencies
├── .env.template             # API key placeholders (copy to .env)
├── .env                      # (Create: your actual API keys)
├── README.md                 # This file
│
├── tools.py                  # Core tool functions
│   ├── load_assignment1_data()
│   ├── generate_data_profile()
│   ├── create_autonomous_charts()
│   ├── select_imputation_strategy()
│   ├── apply_imputation()
│   ├── classify_text_slm_batch()
│   └── compare_llm_vs_slm_classifications()
│
├── agents.py                 # Pydantic state schema + node wrappers
│   └── AgentState (TypedDict with 15+ fields)
│
├── workflow.py               # LangGraph state machine
│   ├── load_data_node
│   ├── validate_gateway
│   ├── eda_analysis_node
│   ├── cleaning_strategy_node
│   ├── report_node
│   └── MemorySaver checkpointing
│
└── app.py                    # Streamlit dashboard
    ├── Real-time workflow streaming
    ├── 4 tabs: Data, EDA, LLM vs SLM, Report
    └── Download cleaned CSV button
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
cd /home/ridhwanlaptop/assignment1_datamining/ridhwan/ai_agent
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the template to your actual `.env` file:

```bash
cp .env.template .env
```

Edit `.env` and fill in your credentials:
- **AZURE_OPENAI_KEY**: Your Azure OpenAI API key
- **AZURE_OPENAI_ENDPOINT**: Your Azure resource endpoint (e.g., https://my-resource.openai.azure.com/)
- **AZURE_OPENAI_DEPLOYMENT_NAME**: Deployment name in Azure (e.g., "gpt-4.1")
- **HUGGINGFACE_TOKEN**: Your HuggingFace API token (for Qwen access)

### 3. Verify Assignment 1 CSV Exists

The application expects the patient CSV at:
```
/home/ridhwanlaptop/assignment1_datamining/ridhwan/artifacts/generated_patients.csv
```

This is automatically generated from `assignment1_main.ipynb`. If missing, re-run that notebook first.

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 UI Walkthrough

### Tab 1: Loaded Data
- First 10 rows of the patient dataset
- **Data Profile** (LLM-generated): Natural-language summary of dataset characteristics
- **Missing Statistics**: Before imputation, shows % missing per column

### Tab 2: EDA Charts
- **Three autonomous charts**:
  1. **Primary Chart** (Autonomous Selection):
     - Scatter plot if age-BP correlation > 0.6
     - Boxplot otherwise
  2. **Heatmap**: Correlation matrix of numerical columns
  3. **Diagnosis Distribution**: Pie chart of diagnosis types
- **Chart Metadata**: Shows decision logic (correlation value, strategy chosen)

### Tab 3: LLM vs SLM Comparison
- **Left Column**: LLM (gpt-4.1) classifications of 20 sample diagnoses
- **Right Column**: SLM (Qwen2.5-1.5B) classifications of same diagnoses
- **Metric**: Agreement rate (%) between models

### Tab 4: Cleaning Report
- **Imputation Summary**: Strategy chosen + decision log
- **Before/After Missingness**: Bar chart showing impact
- **SLM Classification Distribution**: Pie chart of CRITICAL vs NON-CRITICAL

### Sidebar Features
- **Agent Status**: Shows current node running ("Loading Data", "Analyzing EDA", etc.)
- **Live Logs**: Displays autonomous decision reasoning
- **Progress Bar**: Overall workflow completion %

---

## 🤔 How Autonomous Decisions Work

The agent uses a **state machine** (LangGraph) to manage data and decision-making:

1. **State Management**: All intermediate results stored in `AgentState` (Pydantic TypedDict)
   - Dataset
   - Chart metadata (correlation values)
   - Missing data statistics
   - Classification results
   - Logs of decisions made

2. **Checkpointing**: LangGraph's `MemorySaver` allows resuming workflows from any node
   - Useful for debugging or iterating on specific steps

3. **Conditional Routing**: Gateway nodes decide which path to take
   - If data loads → proceed to EDA
   - If load fails → error handler

4. **Logging**: Every autonomous decision logged with rationale
   - Example: "Correlation (age, systolic_bp) = 0.72 > 0.6 → Scatter plot selected"

---

## 🎯 Comparison: Static Notebook vs. Agentic Workflow

| Aspect | Assignment 1 Notebook | Agentic AI Pipeline |
|--------|----------------------|---------------------|
| Data Processing | Hardcoded steps | Dynamic decisions |
| Chart Selection | Predefined (all charts) | Correlation-based selection |
| Imputation | Hardcoded (KNN) | Strategy chosen by agent |
| Transparency | Linear code cells | Decision logs + reasoning |
| Resumability | Rerun entire notebook | Resume from checkpoints |
| Model Deployment | Local notebook | Scalable workflow API |
| Monitoring | Manual notebook review | Real-time Streamlit dashboard |

---

## 📈 Example Workflow Run

### Input
- **CSV**: 500 patient records with ~15% missing values

### Agent Decisions (Autonomous)
1. **Load**: ✓ Valid CSV, 500 records, all columns present
2. **Profile**: LLM generates: "Dataset contains 500 patients with vital signs (HR, BP, temp) and clinical diagnoses. Age ranges 18–85. Missing data concentrated in temperature column (~20%)."
3. **EDA Chart**: Computes age-BP correlation = 0.72 → Selects scatter plot
4. **Imputation**: Detects 15% missing → Selects KNN imputation (K=5)
5. **Classification**: 
   - SLM classifies all 500 diagnoses as CRITICAL/NON-CRITICAL
   - LLM vs SLM agreement on 20 samples: 85%

### Output
- Cleaned DataFrame (no missing values)
- 3 visualization images (embedded in Streamlit)
- Metadata: strategy used, decision rationale, model comparison
- Download: cleaned_patients.csv

---

## 🔧 Troubleshooting

### "CSV not found" error
- Ensure `assignment1_main.ipynb` has been run
- Check file exists: `/home/ridhwanlaptop/assignment1_datamining/ridhwan/artifacts/generated_patients.csv`

### "AZURE_OPENAI_KEY not set" error
- Copy `.env.template` to `.env`
- Edit `.env` with your actual Azure OpenAI credentials
- Restart Streamlit app

### "HuggingFace token invalid" error
- Verify `HUGGINGFACE_TOKEN` in `.env`
- Ensure token has access to Qwen2.5-1.5B-Instruct model
- Regenerate token at https://huggingface.co/settings/tokens

### Streamlit runs but no data loads
- Check logs in sidebar for error messages
- Verify `.env` file is in same directory as `app.py`
- Run `python3 -c "from tools import load_assignment1_data; print('OK')"` to test imports

---

## 📚 References

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **HuggingFace Router API**: https://huggingface.co/inference-api (see "Serverless Inference API")
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

## 📝 Assignment Alignment

This agentic AI pipeline demonstrates all three components of **WQD7005 Assignment 1**:

- **Task 1 (Dataset Simulation – 2 marks)**: Loads pre-generated 500-patient CSV; LLM generates narrative profile
- **Task 2 (EDA with LLM – 3 marks)**: Autonomous chart selection based on correlation; LLM provides trend analysis
- **Task 3 (AI Data Prep – 3 marks)**: Autonomous imputation strategy selection; SLM text classification; LLM vs SLM comparison

**Bonus**: Demonstrates **autonomous decision-making** and **real-time monitoring** for potential extra credit on group project integration.

---

## 🎓 Key Learnings

1. **State Machines**: LangGraph provides a clean abstraction for complex multi-step workflows
2. **Autonomous Decisions**: Even simple rules (correlation threshold, missingness %) can create intelligent agent behavior
3. **Model Orchestration**: Combining LLM (reasoning) + SLM (efficiency) optimizes cost and performance
4. **Checkpointing**: Enabling resumable workflows makes systems more robust and debuggable
5. **Monitoring**: Real-time dashboards (Streamlit) make agent behavior transparent and trustworthy

---

**Happy exploring! 🚀**
