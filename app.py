import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import time
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
    }
    .sample-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .sample-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    .stSlider {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the summarization model from Hugging Face with proper configuration"""
    try:
        model_name = "bariraazaib"
        
        with st.sidebar:
            with st.spinner("🔄 Loading AI Model..."):
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Try different loading strategies
                try:
                    # First try: Load directly
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    st.success("✅ Model Ready!")
                    
                except Exception as e:
                    st.warning("⚠️ Loading alternative configuration...")
                    
                    # Second try: Load base model + LoRA
                    base_model_name = "facebook/bart-base"
                    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                    model = PeftModel.from_pretrained(model, model_name)
                    st.success("✅ Model Ready!")
                
                # Move to device if not using device_map
                if not hasattr(model, 'hf_device_map'):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)
                
                return tokenizer, model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        return None, None

def generate_summary(tokenizer, model, text, max_length=128, min_length=30):
    """Generate summary with proper parameters"""
    try:
        # Device detection
        if hasattr(model, 'hf_device_map'):
            device = next(iter(model.hf_device_map.values()))
        else:
            device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(
            text, 
            max_length=512, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        # Clear cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                temperature=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up
        del inputs, summary_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return summary
        
    except Exception as e:
        st.error(f"⚠️ Generation Error: {str(e)}")
        return None

def main():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<h1>✨ AI Text Summarizer</h1>", unsafe_allow_html=True)
        st.markdown("**Powered by** `mustehsannisarrao/summarizer` | Advanced NLP Model")
    
    st.markdown("---")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("⚠️ Failed to load model. Please refresh the page.")
        return
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📝 Input Text")
        text_input = st.text_area(
            "",
            height=250,
            placeholder="Paste your text here to generate an intelligent summary...",
            value="The company reported strong earnings this quarter with profits increasing by 20%. This growth was driven by successful product launches and expanding market share in emerging economies. The CEO expressed optimism about future growth prospects and announced plans for further international expansion.",
            label_visibility="collapsed"
        )
        
        # Parameter controls
        st.markdown("### ⚙️ Generation Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_length = st.slider("📏 Max Length", 50, 200, 128, help="Maximum summary length")
        with col2:
            min_length = st.slider("📐 Min Length", 10, 100, 30, help="Minimum summary length")
        with col3:
            num_beams = st.selectbox("🔍 Beam Size", [2, 4, 6], index=1, help="Higher = better quality, slower")
        
        # Action buttons
        st.markdown("###")
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        
        with col_btn1:
            generate_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)
        
        with col_btn2:
            debug_btn = st.button("🔧 Debug Mode", use_container_width=True)
    
    with col_right:
        st.markdown("### 📊 Quick Stats")
        
        # Model Info Card
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">🤖 Model</h3>
                <p style="font-size: 14px; color: #666;">BART-based Seq2Seq</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("###")
        
        # Input Stats
        char_count = len(text_input)
        word_count = len(text_input.split())
        
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #764ba2; margin: 0;">{word_count}</h3>
                <p style="font-size: 14px; color: #666;">Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">{char_count}</h3>
                <p style="font-size: 14px; color: #666;">Characters</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate Summary
    if generate_btn:
        if text_input.strip():
            with st.spinner("🔮 Generating your summary..."):
                start_time = time.time()
                summary = generate_summary(tokenizer, model, text_input, max_length, min_length)
                end_time = time.time()
                
                if summary:
                    st.markdown("### 📄 Generated Summary")
                    st.markdown(f"""
                        <div class="summary-box">
                            <h3 style="margin-top: 0;">Summary</h3>
                            <p style="font-size: 18px; line-height: 1.6;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Processing Time", f"{(end_time - start_time):.2f}s")
                    with col2:
                        st.metric("📊 Summary Length", f"{len(summary)} chars")
                    with col3:
                        compression = round((1 - len(summary)/len(text_input)) * 100, 1)
                        st.metric("🗜️ Compression", f"{compression}%")
        else:
            st.warning("⚠️ Please enter some text to summarize.")
    
    if debug_btn:
        if text_input.strip():
            with st.spinner("🔍 Running diagnostics..."):
                st.code(f"Input Length: {len(text_input)} characters\nModel Device: {next(model.parameters()).device}\nTokenizer: {tokenizer.__class__.__name__}")
    
    # Test Samples Section
    st.markdown("---")
    st.markdown("### 🎯 Try Sample Texts")
    
    test_samples = {
        "📰 Tech News": "Researchers discovered a new species of marine life in the deep ocean. The creature has unique bioluminescent properties that help it communicate in complete darkness. This discovery could lead to advances in biomedical imaging technology.",
        "📱 Product Review": "The new smartphone features a revolutionary camera system that outperforms all competitors. With advanced AI processing and enhanced low-light capabilities, it sets a new standard for mobile photography.",
        "🌍 Climate Report": "Climate change continues to affect global weather patterns, with scientists reporting increased frequency of extreme weather events. Governments worldwide are implementing new policies to address this growing concern."
    }
    
    cols = st.columns(3)
    for i, (label, sample) in enumerate(test_samples.items()):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"sample_{i}"):
                st.session_state.test_text = sample
                st.rerun()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This AI-powered summarizer uses state-of-the-art 
        transformer models to generate concise summaries 
        of long text passages.
        
        **Features:**
        - Fast processing
        - Adjustable parameters
        - Multiple beam search
        - GPU acceleration
        """)
        
        st.markdown("### 🎯 Tips")
        st.success("""
        - Keep input text between 100-1000 words
        - Adjust beam size for quality vs speed
        - Use debug mode to troubleshoot
        """)

if __name__ == "__main__":
    main()
