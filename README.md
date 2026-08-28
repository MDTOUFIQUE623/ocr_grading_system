# Answer AI OCR & Grading System

An AI-powered system that extracts text from handwritten answer sheets using Vision-Language Models (VLM) and traditional OCR, then automatically evaluates and grades student responses against reference rubrics.

## Features

- **Dual OCR Engines**: 
  - **VLM Engine**: Powered by Google Gemini 2.5 Flash for high-accuracy handwriting extraction.
  - **Traditional OCR Engine**: Powered by EasyOCR for text block localization and baseline comparison.
- **Image Preprocessing**: Enhances images (contrast adjustment, denoising) for higher OCR accuracy.
- **Automated Grading**: Uses Gemini to evaluate conceptual accuracy, assign scores out of 5, and generate constructive feedback.
- **Interactive Web UI**: Streamlit dashboard to upload answers, compare engine similarity metrics, and view evaluation results.
- **Feedback Loop**: Records user corrections into `feedback.jsonl` for evaluation.

## Prerequisites

- Python >= 3.12
- Google Gemini API Key

## Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Streamlit Web App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

### CLI Test Run
Execute pipeline on a sample image:
```bash
python test_run.py
```

## Project Structure

- [app.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/app.py): Streamlit UI dashboard.
- [vlm_ocr_engine.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/vlm_ocr_engine.py): VLM text extraction via Gemini 2.5 Flash.
- [ocr_engine.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/ocr_engine.py): Traditional OCR via EasyOCR.
- [evaluator.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/evaluator.py): LLM answer grading and feedback generation.
- [image_preprocessing.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/image_preprocessing.py): Contrast enhancement and image processing.
- [feedback.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/feedback.py): Logging corrections to `feedback.jsonl`.
- [test_run.py](file:///c:/Users/Lenovo/Downloads/python%20projects/answer_ocr/answer_ai_ocr/test_run.py): End-to-end command line test script.
