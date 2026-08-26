import streamlit as st
import os
import uuid
from difflib import SequenceMatcher
import vlm_ocr_engine
import ocr_engine
import evaluator
import feedback
import image_preprocessing

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

st.title("OCR Grading System")

uploaded_file = st.file_uploader("Upload handwritten answer", type=["png", "jpg", "jpeg"])
reference_text = st.text_area("Reference Answer / Rubric (Leave blank to auto-generate for demo)")

if uploaded_file:
    if st.button("Process & Grade"):
        # ponytail: short UUID for readable filenames, and sanitize to fix OpenCV unicode bugs
        img_id = uuid.uuid4().hex[:8]
        import re
        safe_name = re.sub(r'[^\w\.-]', '_', uploaded_file.name)
        in_path = os.path.join("input", f"{img_id}_{safe_name}")
        out_dir = os.path.join("output", img_id)
        
        with open(in_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.session_state.img_id = img_id
        st.session_state.in_path = in_path
        st.session_state.reference = reference_text
        
        with st.spinner("Preprocessing & Extracting text..."):
            image_preprocessing.preprocess(in_path, output_dir=out_dir)
            
            # VLM Engine
            extracted = vlm_ocr_engine.extract_text_with_vlm(in_path)
            st.session_state.extracted = extracted
            
            # Traditional Engine
            ocr_blocks = ocr_engine.process_image(in_path)
            st.session_state.ocr_extracted = "\n".join([b["text"] for b in ocr_blocks])

            
        with st.spinner("Preparing Reference..."):
            if not reference_text.strip():
                ref = evaluator.generate_demo_reference(extracted)
            else:
                ref = reference_text
            st.session_state.reference = ref
            
        with st.spinner("Evaluating..."):
            result = evaluator.evaluate_answer(extracted, st.session_state.reference)
            st.session_state.result = result

if "result" in st.session_state:
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.in_path, caption=f"ID: {st.session_state.img_id}")
    with col2:
        st.write(f"**Score:** {st.session_state.result.get('score')} / 5")
        st.write(f"**Feedback:** {st.session_state.result.get('feedback')}")
        st.write("**Extracted Text:**")
        st.text(st.session_state.extracted)
        st.write("**Reference Used:**")
        st.text(st.session_state.reference)
    
    st.divider()
    st.subheader("System Feedback Loop")
    with st.form("feedback_form"):
        correction = st.text_input("Corrections (e.g., 'Score should be 4', 'Missed word X')")
        if st.form_submit_button("Submit to Feedback Loop"):
            record = {
                "image_id": st.session_state.img_id,
                "extracted_text": st.session_state.extracted,
                "reference": st.session_state.reference,
                "score": st.session_state.result.get('score'),
                "llm_feedback": st.session_state.result.get('feedback'),
                "user_correction": correction
            }
            feedback.save_feedback(record)
            st.success("Saved to feedback.jsonl!")
            
    st.divider()
    st.subheader("OCR Engine Comparison")
    
    vlm_text = st.session_state.extracted
    ocr_text = st.session_state.ocr_extracted
    
    # ponytail: simple ratio from stdlib
    similarity = SequenceMatcher(None, vlm_text, ocr_text).ratio()
    
    st.metric("VLM vs Traditional OCR Similarity", f"{similarity:.1%}")
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.write("**VLM Engine (Gemini)**")
        st.info(vlm_text)
    with comp_col2:
        st.write("**Traditional Engine (Paddle + TrOCR)**")
        st.warning(ocr_text)
