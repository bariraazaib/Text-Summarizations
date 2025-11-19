# app_fixed.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import time
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the summarization model from Hugging Face with proper configuration"""
    try:
        model_name = "mustehsannisarrao/summarizer"
        
        st.sidebar.info("🔄 Loading model... This may take a minute.")
        
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
            st.sidebar.success("✅ Model loaded directly")
            
        except Exception as e:
            st.sidebar.warning(f"Direct load failed: {e}. Trying alternative...")
            
            # Second try: Load base model + LoRA
            base_model_name = "facebook/bart-base"
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(model, model_name)
            st.sidebar.success("✅ Model loaded with LoRA adapters")
        
        # Move to device if not using device_map
        if not hasattr(model, 'hf_device_map'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            st.sidebar.info(f"📱 Model moved to: {device}")
        
        return tokenizer, model
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
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
        
        # Generate summary with different parameters
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,  # Changed from 8 to 4 for faster generation
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,  # Reduced from 2.0
                temperature=0.9,     # Added temperature for variation
                do_sample=False,     # Beam search for consistency
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
        st.error(f"Error generating summary: {str(e)}")
        return None

def debug_generation(tokenizer, model, text):
    """Debug function to see what's happening during generation"""
    try:
        device = next(model.parameters()).device
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        st.write("🔍 Debug Info:")
        st.write(f"Input shape: {inputs.input_ids.shape}")
        st.write(f"Device: {device}")
        
        # Test with simpler generation
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                min_length=20,
                num_beams=2,  # Simpler beam search
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        return f"Debug error: {e}"

def main():
    st.title("📝 Paragraph Summarizer")
    st.markdown("Using model: `mustehsannisarrao/summarizer`")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Input section
    st.subheader("Enter text to summarize:")
    
    text_input = st.text_area(
        "Text:",
        height=200,
        placeholder="Paste your text here...",
        value="The company reported strong earnings this quarter with profits increasing by 20%. This growth was driven by successful product launches and expanding market share in emerging economies. The CEO expressed optimism about future growth prospects and announced plans for further international expansion."
    )
    
    # Generation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("Max Length", 50, 200, 128)
    with col2:
        min_length = st.slider("Min Length", 10, 100, 30)
    with col3:
        num_beams = st.selectbox("Beam Size", [2, 4, 6], index=1)
    
    # Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Generating summary..."):
                    start_time = time.time()
                    summary = generate_summary(tokenizer, model, text_input, max_length, min_length)
                    end_time = time.time()
                    
                    if summary:
                        st.subheader("📄 Generated Summary:")
                        st.success(summary)
                        
                        # Metrics
                        st.metric("Processing Time", f"{(end_time - start_time):.2f}s")
                        st.metric("Summary Length", f"{len(summary)} characters")
                        
                        # Check if summary is repetitive
                        if "The company reported strong earnings" in summary and len(summary) > 50:
                            st.warning("⚠️ The summary might be repetitive. Try adjusting generation parameters.")
    
    with col2:
        if st.button("🐛 Debug Generation", use_container_width=True):
            if text_input.strip():
                with st.spinner("Running debug..."):
                    debug_result = debug_generation(tokenizer, model, text_input)
                    st.subheader("Debug Result:")
                    st.code(debug_result)
    
    # Test with different texts
    st.subheader("Quick Test Samples:")
    test_samples = [
        "Researchers discovered a new species of marine life in the deep ocean. The creature has unique bioluminescent properties that help it communicate in complete darkness. This discovery could lead to advances in biomedical imaging technology.",
        "The new smartphone features a revolutionary camera system that outperforms all competitors. With advanced AI processing and enhanced low-light capabilities, it sets a new standard for mobile photography.",
        "Climate change continues to affect global weather patterns, with scientists reporting increased frequency of extreme weather events. Governments worldwide are implementing new policies to address this growing concern."
    ]
    
    for i, sample in enumerate(test_samples):
        if st.button(f"Test Sample {i+1}", key=f"sample_{i}"):
            st.session_state.test_text = sample
            st.rerun()

if __name__ == "__main__":
    main()
