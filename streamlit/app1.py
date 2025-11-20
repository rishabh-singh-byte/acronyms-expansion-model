# app/streamlit/app1.py
"""
Multi-query evaluation Streamlit interface for model comparison.
Samples random queries from dataset and compares responses across multiple AI models.
Requires FastAPI server running on port 8090.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time

API_URL = "http://localhost:8090/inference/generate"

st.set_page_config(
    page_title="Acronym Expansion Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .model-result {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .query-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px 8px 0 0;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def format_model_name(model_name: str) -> str:
    """Format model name for display"""
    return model_name.replace('_', ' ').title()

def render_candidate_acronyms(acronyms_data: str) -> None:
    """Render candidate acronyms in styled boxes"""
    if not acronyms_data:
        st.info("No candidate acronyms available")
        return
    
    acronyms_dict = {}
    
    if isinstance(acronyms_data, str):
        # Try parsing as JSON first
        try:
            acronyms_dict = json.loads(acronyms_data)
        except json.JSONDecodeError:
            import re
            cleaned = acronyms_data.strip().strip('()').strip()
            if cleaned:
                pattern = r'([^:,]+):\s*([^,()]+)'
                matches = re.findall(pattern, cleaned)
                if matches:
                    for acronym, expansion in matches:
                        acronym_key = acronym.strip().upper()
                        expansion_value = expansion.strip()
                        if acronym_key not in acronyms_dict:
                            acronyms_dict[acronym_key] = []
                        acronyms_dict[acronym_key].append(expansion_value)
                else:
                    st.markdown(f"""
                    <div style="
                        background-color: #e3f2fd; 
                        color: #1976d2; 
                        padding: 1rem; 
                        border-radius: 12px; 
                        margin-bottom: 0.8rem; 
                        border: 1px solid #bbdefb;
                    ">
                        {acronyms_data}
                    </div>
                    """, unsafe_allow_html=True)
                    return
    elif isinstance(acronyms_data, dict):
        acronyms_dict = acronyms_data
    else:
        st.info("Invalid acronyms format")
        return
    
    if not acronyms_dict:
        st.info("No candidate acronyms available")
        return
    
    for acronym, expansions in acronyms_dict.items():
        if not expansions:
            expansions = ["No expansions available"]
        st.markdown(f"""
        <div style="
            background-color: #e3f2fd; 
            color: #1976d2; 
            padding: 1rem; 
            border-radius: 12px; 
            margin-bottom: 0.8rem; 
            border: 1px solid #bbdefb;
        ">
            <strong style='font-size: 1.1rem;'>{acronym}</strong>
            <ul style='margin-top: 0.5rem;'>
        """, unsafe_allow_html=True)
        
        for expansion in expansions:
            st.markdown(f"<li>{expansion}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)


def parse_model_output(output: Any) -> Dict[str, List[str]]:
    """Parse and normalize model output to dict format"""
    if isinstance(output, dict):
        return output
    
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = eval(output)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    
    return {"response": [str(output)]}

def render_model_results(results: Dict[str, Any]) -> None:
    """Display model results in columnar layout"""
    if not results:
        st.info("No model results available")
        return
    
    cols = st.columns(len(results))
    
    for idx, (model_name, output) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="model-result">
                <h4 style="color: #667eea; margin-bottom: 1rem;">
                    🤖 {format_model_name(model_name)}
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            parsed_output = parse_model_output(output)
            
            for key, values in parsed_output.items():
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                if isinstance(values, list):
                    for value in values:
                        st.markdown(f"• {value}")
                else:
                    st.markdown(f"• {values}")
                st.markdown("")

st.markdown(
    '<h1 class="main-header">🤖 Acronym Expansion Assistant</h1>',
    unsafe_allow_html=True
)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Generate and evaluate acronym expansions using multiple AI models
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔧 Configuration")
    
    st.markdown("### 🤖 Model Selection")
    
    use_qwen_base = st.checkbox(
        "🔵 Qwen Base Model", 
        value=True,
        help="Use the base Qwen model for acronym expansion"
    )
    use_qwen_lora = st.checkbox(
        "🟡 Qwen LoRA Adapter", 
        value=True,
        help="Use the fine-tuned Qwen model with LoRA adapter"
    )
    use_openai_gpt = st.checkbox(
        "🟢 OpenAI GPT", 
        value=True,
        help="Use OpenAI's GPT model for comparison"
    )
    use_tiny_llama_lora = st.checkbox(
        "🟠 TinyLlama LoRA Adapter", 
        value=False,
        help="Use the fine-tuned TinyLlama model with LoRA adapter"
    )
    
    selected_models = sum([use_qwen_base, use_qwen_lora, use_openai_gpt, use_tiny_llama_lora])
    if selected_models == 0:
        st.error("⚠️ Please select at least one model!")
    
    st.markdown("---")
    
    st.markdown("### 📊 Evaluation Settings")
    n_samples = st.number_input(
        "Number of Random Samples", 
        min_value=1, 
        max_value=50, 
        value=3, 
        step=1,
        help="Number of random queries to evaluate"
    )
    
    st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 🎲 Generate Random Acronym Queries")
    st.markdown("Evaluate acronym expansion performance across multiple AI models")

with col2:
    if selected_models > 0:
        st.metric("Selected Models", selected_models, delta=None)
    else:
        st.error("No models selected")

st.markdown("---")

run_button_col1, run_button_col2, run_button_col3 = st.columns([1, 2, 1])
with run_button_col2:
    run_evaluation = st.button(
        "🚀 Run Evaluation", 
        use_container_width=True,
        disabled=(selected_models == 0),
        type="primary"
    )

if run_evaluation:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    with st.spinner("🔄 Processing evaluation request..."):
        try:
            status_text.text("📡 Sending request to API...")
            progress_bar.progress(20)
            
            response = requests.post(
                API_URL,
                json={
                    "n": n_samples,
                    "use_qwen_base": use_qwen_base,
                    "use_qwen_lora": use_qwen_lora,
                    "use_openai_gpt": use_openai_gpt,
                    "use_tiny_llama_lora": use_tiny_llama_lora
                },
                timeout=120
            )
            
            progress_bar.progress(60)
            status_text.text("🧠 Models are generating responses...")

            if response.status_code == 200:
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                data = response.json()
                total = data.get("total_samples", 0)
                all_results = data.get("data", [])
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Evaluation Complete! Processed {total} queries in {elapsed_time:.2f} seconds.")
                st.markdown("### 🔍 Detailed Results")
                
                for idx, item in enumerate(all_results, start=1):
                    query = item.get('query', '')
                    candidate_acronyms = item.get('candidate_acronyms', '')
                    results = item.get("results", {})
                    
                    with st.expander(f"📋 Query {idx}: {query}", expanded=(idx == 1)):
                        # Query header with styling
                        st.markdown(f"""
                        <div class="query-header">
                            <h4 style="margin: 0; color: white;">🔍 Query: {query}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        render_candidate_acronyms(candidate_acronyms)
                        st.markdown("")
                        st.markdown("**🧠 Model Responses:**")
                        render_model_results(results)

            else:
                progress_bar.empty()
                status_text.empty()
                
                st.markdown(f"""
                <div class="error-message">
                    <h4>❌ API Error</h4>
                    <p><strong>Status Code:</strong> {response.status_code}</p>
                    <p><strong>Response:</strong> {response.text}</p>
                </div>
                """, unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.empty()
            st.markdown("""
            <div class="error-message">
                <h4>⏰ Request Timeout</h4>
                <p>The request took too long to complete. Please try again with fewer samples.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            status_text.empty()
            st.markdown("""
            <div class="error-message">
                <h4>🔌 Connection Error</h4>
                <p>Could not connect to the API server. Please check if the server is running.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.markdown(f"""
            <div class="error-message">
                <h4>🚨 Request Failed</h4>
                <p><strong>Error:</strong> {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9em;">
    Built using Streamlit
</div>
""", unsafe_allow_html=True)
