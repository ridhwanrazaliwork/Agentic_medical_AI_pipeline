"""
AI Agent Tools for Medical Data Analysis Pipeline
- Data loading & validation
- EDA (charts with autonomous selection)
- Imputation strategy selection
- SLM text classification via local HuggingFace transformers (Qwen2.5-0.5B)
"""

import os
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables - explicit path
current_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(current_dir, ".env")
load_dotenv(env_file)

# Initialize clients
def get_azure_client():
    """Initialize Azure OpenAI client"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    if not endpoint or not api_key:
        raise ValueError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY not in .env")
    return OpenAI(
        base_url=f"{endpoint}/openai/v1",
        api_key=api_key
    )

def get_hf_client():
    """Initialize HuggingFace Router client for SLM (Meta Llama via HF)"""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not in .env (for Meta Llama inference)")
    
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

# ===================== UTILITY FUNCTIONS =====================

def to_native_python(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types (for msgpack serialization)
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native_python(item) for item in obj]
    return obj

# ===================== TASK 1: DATA LOADING =====================

def load_assignment1_data(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load pre-generated patient data from assignment1.ipynb CSV
    
    Args:
        csv_path: Path to generated_patients.csv from assignment1
        
    Returns:
        (df, metadata) where metadata includes validation info
        
    Raises:
        FileNotFoundError: If CSV not found
        ValueError: If schema mismatch or row count wrong
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ CSV not found: {csv_path}\n\n"
            f"Make sure assignment1.ipynb has run successfully.\n"
            f"Check that generated_patients.csv exists at the path above.\n"
            f"Expected path:\n  /home/ridhwanlaptop/assignment1_datamining/ridhwan/artifacts/generated_patients.csv"
        )
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"❌ Failed to read CSV: {str(e)}")
    
    # Validate schema (flexible - just check key columns exist)
    required_cols = {
        'patient_id', 'age', 'gender', 'visit_date', 
        'vital_heart_rate', 'vital_systolic_bp', 'vital_diastolic_bp',
        'vital_temperature', 'diagnosis', 'symptoms'
    }
    actual_cols = set(df.columns)
    missing_cols = required_cols - actual_cols
    
    if missing_cols:
        raise ValueError(
            f"❌ Schema mismatch!\n\n"
            f"Expected columns: {sorted(required_cols)}\n"
            f"Actual columns:   {sorted(actual_cols)}\n"
            f"Missing:          {sorted(missing_cols)}"
        )
    
    if len(df) != 500:
        raise ValueError(
            f"❌ Row count mismatch!\n"
            f"Expected 500 rows, got {len(df)}"
        )
    
    # Log successful load
    metadata = {
        "status": "loaded",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "data_source": "GenAI-generated from assignment1.ipynb",
        "loaded_from": csv_path,
        "timestamp": datetime.now().isoformat(),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_percent": float(round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2))
    }
    
    # Convert to native Python types for msgpack serialization
    metadata = to_native_python(metadata)
    
    return df, metadata


def generate_data_profile(df: pd.DataFrame, client: OpenAI, deployment: str) -> str:
    """
    Generate natural language summary of dataset using LLM (Task 1 aspect)
    Minimal API calls (1 per workflow)
    
    Args:
        df: Patient DataFrame
        client: Azure OpenAI client
        deployment: Deployment name (e.g., 'gpt-4.1')
        
    Returns:
        Natural language profile summary
    """
    # Calculate key statistics
    age_vals = pd.to_numeric(df['age'], errors='coerce').dropna()
    hr_vals = pd.to_numeric(df['vital_heart_rate'], errors='coerce').dropna()
    sbp_vals = pd.to_numeric(df['vital_systolic_bp'], errors='coerce').dropna()
    
    missing_summary = df.isnull().sum().sum()
    missing_pct = (missing_summary / (len(df) * len(df.columns))) * 100
    
    gender_dist = df['gender'].value_counts().to_dict()
    top_diagnosis = df['diagnosis'].value_counts().head(1)
    top_diag_name = top_diagnosis.index[0] if len(top_diagnosis) > 0 else "Unknown"
    top_diag_count = top_diagnosis.values[0] if len(top_diagnosis) > 0 else 0
    
    stats_text = f"""
Dataset Statistics (500 Malaysian patients):
- Mean age: {age_vals.mean():.1f} years (SD: {age_vals.std():.1f})
- Gender distribution: {gender_dist}
- Mean heart rate: {hr_vals.mean():.0f} bpm
- Mean systolic BP: {sbp_vals.mean():.0f} mmHg
- Missing data: {missing_summary} points ({missing_pct:.1f}%)
- Top diagnosis: {top_diag_name} ({top_diag_count} cases)
- Total features: {len(df.columns)}
"""
    
    # LLM generates profile (concise, 3-4 sentences)
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a healthcare data analyst. Be concise and factual."
                },
                {
                    "role": "user",
                    "content": f"Generate a 3-4 sentence profile of this patient dataset:\n{stats_text}"
                }
            ],
            temperature=0.7,
            max_completion_tokens=200
        )
        profile = response.choices[0].message.content
    except Exception as e:
        profile = f"[Profile generation error: {str(e)}]"
    
    return profile


# ===================== TASK 2: EDA =====================

def create_autonomous_charts(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    """
    Create 2-3 charts with AUTONOMOUS selection based on data patterns
    
    Decision Logic:
    - Chart 1: Age vs Systolic BP: if |correlation| > 0.6 → scatter with trendline, else → boxplot
    - Chart 2: Always correlation heatmap of vital signs
    - Chart 3: Diagnosis distribution bar chart
    
    Returns:
        (charts_base64, metadata) where charts are base64-encoded PNG images
    """
    charts_b64 = []
    metadata = {}
    
    try:
        # ========== DECISION 1: Age vs SBP correlation-based chart type ==========
        age_vals = pd.to_numeric(df['age'], errors='coerce')
        sbp_vals = pd.to_numeric(df['vital_systolic_bp'], errors='coerce')
        
        valid_idx = age_vals.dropna().index.intersection(sbp_vals.dropna().index)
        if len(valid_idx) > 2:
            age_clean = df.loc[valid_idx, 'age'].astype(float)
            sbp_clean = df.loc[valid_idx, 'vital_systolic_bp'].astype(float)
            corr, _ = pearsonr(age_clean, sbp_clean)
            metadata["age_sbp_correlation"] = round(corr, 3)
        else:
            corr = 0
            metadata["age_sbp_correlation"] = 0
        
        # AUTONOMOUS DECISION LOG
        if abs(corr) > 0.6:
            chart_type = "scatter_with_trendline"
            metadata["chart_1_decision"] = f"Strong correlation ({corr:.3f}) → ScatterPlot with trend"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(age_clean, sbp_clean, alpha=0.6, s=50, color='steelblue')
            
            # Add trendline
            z = np.polyfit(age_clean, sbp_clean, 1)
            p = np.poly1d(z)
            ax.plot(age_clean.sort_values(), p(age_clean.sort_values()), 
                   "r--", linewidth=2, label=f"Trend (r={corr:.2f})")
            
            ax.set_xlabel("Age (years)", fontsize=11)
            ax.set_ylabel("Systolic BP (mmHg)", fontsize=11)
            ax.set_title(f"Age vs Systolic BP (Correlation: {corr:.3f})", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            chart_type = "boxplot"
            metadata["chart_1_decision"] = f"Weak correlation ({corr:.3f}) → BoxPlot"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(sbp_clean, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='skyblue', alpha=0.7))
            ax.set_ylabel("Systolic BP (mmHg)", fontsize=11)
            ax.set_title("Systolic BP Distribution", fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart1_b64 = fig_to_base64(fig)
        charts_b64.append(chart1_b64)
        plt.close()
        
        # ========== CHART 2: Correlation heatmap ==========
        vital_cols = [col for col in df.columns if col.startswith('vital_')]
        df_vitals = df[vital_cols].apply(pd.to_numeric, errors='coerce')
        
        if len(vital_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_vitals.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title("Vital Signs Correlation Matrix", fontsize=12, fontweight='bold')
            plt.tight_layout()
            chart2_b64 = fig_to_base64(fig)
            charts_b64.append(chart2_b64)
            metadata["chart_2"] = "Correlation heatmap"
            metadata["correlation_matrix"] = to_native_python(corr_matrix.to_dict())
            plt.close()
        
        # ========== CHART 3: Top diagnoses ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        top_diag = df['diagnosis'].value_counts().head(8)
        top_diag.plot(kind='barh', ax=ax, color='mediumseagreen', alpha=0.7)
        ax.set_xlabel("Number of Cases", fontsize=11)
        ax.set_title("Top 8 Diagnoses in Cohort", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        chart3_b64 = fig_to_base64(fig)
        charts_b64.append(chart3_b64)
        metadata["chart_3"] = "Top diagnoses distribution"
        plt.close()
        
    except Exception as e:
        metadata["chart_error"] = str(e)
    
    # Convert all numpy types to native Python for msgpack serialization
    metadata = to_native_python(metadata)
    
    return charts_b64, metadata


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ===================== TASK 3: DATA CLEANING =====================

def select_imputation_strategy(df: pd.DataFrame) -> Tuple[str, str]:
    """
    AUTONOMOUS DECISION: Select imputation strategy based on missingness
    
    Logic:
    - <5% missing → mean imputation
    - 5-20% missing → KNN imputation (k=5)
    - >20% missing → MICE imputation (iterative)
    
    Returns:
        (strategy_name, decision_log)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_pct = (df[numeric_cols].isnull().sum().sum() / 
                  (len(df) * len(numeric_cols))) * 100
    
    # AUTONOMOUS DECISION
    if missing_pct < 5:
        strategy = "mean_imputation"
        decision = f"Detected {missing_pct:.1f}% missing → Selected MEAN imputation (simple, sufficient for <5%)"
    elif 5 <= missing_pct <= 20:
        strategy = "knn_imputation"
        decision = f"Detected {missing_pct:.1f}% missing → Selected KNN imputation (context-aware, handles 5-20%)"
    else:
        strategy = "mice_imputation"
        decision = f"Detected {missing_pct:.1f}% missing → Selected MICE imputation (advanced, handles >20%)"
    
    return strategy, decision


def apply_imputation(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Apply chosen imputation strategy to numeric columns
    
    Args:
        df: Input DataFrame with missing values
        strategy: 'mean_imputation', 'knn_imputation', or 'mice_imputation'
        
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return df_imputed
    
    try:
        if strategy == "mean_imputation":
            imputer = SimpleImputer(strategy='mean')
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        elif strategy == "knn_imputation":
            imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        elif strategy == "mice_imputation":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    except Exception as e:
        print(f"⚠️ Imputation error: {str(e)}")
        # Fallback to mean
        imputer = SimpleImputer(strategy='mean')
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_imputed


def classify_text_slm_batch(texts: List[str], model: Any, tokenizer: Any) -> List[Dict]:
    """
    Batch classify text using local Qwen2.5-0.5B-Instruct model
    
    Args:
        texts: List of text strings to classify
        model: HuggingFace transformers AutoModel
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of dicts with {text, classification, confidence}
    """
    classifications = []
    
    for i, text in enumerate(texts):
        if not text or pd.isna(text):
            classifications.append({
                "text": str(text)[:50],
                "classification": "Unknown",
                "confidence": 0.0,
                "model": "Qwen2.5-0.5B"
            })
            continue
        
        try:
            classification_prompt = f"""
Classify the following medical complaint/diagnosis into ONE category:
Categories: CRITICAL (emergency, severe), NON-CRITICAL (routine, mild)

Text: {text}

Return ONLY: "CRITICAL" or "NON-CRITICAL"
"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical triage classifier. Respond with only the category."
                },
                {
                    "role": "user",
                    "content": classification_prompt
                }
            ]
            
            # Apply chat template
            text_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
            
            # Generate
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=30,
                temperature=0.9,
                top_p=0.95,
            )
            
            # Remove input tokens
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode
            result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Parse result
            if "CRITICAL" in result.upper():
                classification = "CRITICAL"
                confidence = 0.95
            elif "NON-CRITICAL" in result.upper():
                classification = "NON-CRITICAL"
                confidence = 0.95
            else:
                classification = "UNKNOWN"
                confidence = 0.0
            
            classifications.append({
                "text": str(text)[:50],
                "classification": classification,
                "confidence": confidence,
                "model": "Qwen2.5-0.5B",
                "raw_output": result[:100]  # Debug: show raw output
            })
            
        except Exception as e:
            classifications.append({
                "text": str(text)[:50],
                "classification": "ERROR",
                "confidence": 0.0,
                "model": "Qwen2.5-0.5B",
                "error": str(e)[:100]
            })
    
    return classifications


def compare_llm_vs_slm_classifications(diagnoses: List[str], 
                                     llm_client: OpenAI, 
                                     model: Any,
                                     tokenizer: Any,
                                     llm_deployment: str) -> Dict:
    """
    Compare LLM (Azure GPT-4.1) vs SLM (local Qwen via transformers) classifications
    
    Returns:
        {
            "llm_classifications": [...],
            "slm_classifications": [...],
            "agreement_rate": float,
            "comparison": [...]
        }
    """
    # Sample 10-20 diagnoses for comparison (to avoid too many LLM API calls)
    sample_size = min(20, len(set(diagnoses)))
    sample_diags = list(set(diagnoses))[:sample_size]
    
    llm_results = []
    slm_results = []
    
    # LLM classification (via Azure)
    for diag in sample_diags:
        try:
            response = llm_client.chat.completions.create(
                model=llm_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Classify into CRITICAL or NON-CRITICAL. Return only the category."
                    },
                    {
                        "role": "user",
                        "content": f"Classify: {diag}"
                    }
                ],
                temperature=0.3,
                max_completion_tokens=30
            )
            llm_class = response.choices[0].message.content.strip()
            llm_results.append({"diagnosis": diag, "classification": llm_class})
        except Exception as e:
            llm_results.append({"diagnosis": diag, "classification": "ERROR", "error": str(e)})
    
    # SLM classification (local model)
    slm_results = classify_text_slm_batch(sample_diags, model, tokenizer)
    
    # Calculate agreement
    agreement_count = 0
    for llm_res, slm_res in zip(llm_results, slm_results):
        if llm_res.get("classification", "").upper() == slm_res.get("classification", "").upper():
            agreement_count += 1
    
    agreement_rate = (agreement_count / len(sample_diags)) * 100 if sample_diags else 0
    
    return {
        "llm_classifications": llm_results,
        "slm_classifications": slm_results,
        "agreement_rate": round(agreement_rate, 1),
        "sample_size": len(sample_diags),
        "comparison_note": "Comparing Azure GPT-4.1 (LLM) vs Qwen2.5-0.5B (SLM via local transformers) classification accuracy"
    }
