Resume Category Predictor using BERT

This project is an NLP-based Resume Classification System that automatically predicts the job category of a given resume using BERT (Bidirectional Encoder Representations from Transformers) embeddings and Logistic Regression.It supports both text input and PDF uploads, making it easy to analyze resumes in different formats.

Features

Accepts resume text or PDF file as input
Utilizes BERT embeddings to capture contextual meaning from resumes
Classifies resumes into 24 professional categories (e.g., Engineering, Finance, Healthcare, IT, Teaching, etc.)
Interactive Gradio web interface with clean UI
Trained using the Kaggle Resume Dataset

Technologies Used

Python, Pandas, Scikit-learn
BERT (Hugging Face Transformers)
PyTorch
Gradio for UI
PyPDF2 for PDF text extraction

Model Details

Text preprocessed and converted into BERT embeddings
Classified using Logistic Regression
Achieved ~76% accuracy on test data
Fine-tuned to correctly identify 24 job categories such as HR, IT, Teacher, Banking, Designer, etc.

Future Improvements

Deploy as a web app (Streamlit/Hugging Face Spaces)
Add support for multilingual resumes
Improve accuracy using fine-tuned domain-specific BERT models
