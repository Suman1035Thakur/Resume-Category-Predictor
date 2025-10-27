#  Resume Category Predictor

A Natural Language Processing (NLP)-based project that automatically classifies resumes into their respective job categories using **BERT embeddings** and **Logistic Regression**.  
The model supports both **text input** and **PDF uploads**, allowing real-time predictions through an interactive Gradio interface.

---

##  Project Overview

This project aims to build an **intelligent resume classification system** that analyzes the textual content of a resume and predicts the most relevant job category (e.g., *Engineering, Healthcare, HR, Finance*, etc.).  
The system uses **advanced NLP techniques** such as text preprocessing, BERT embeddings, and machine learning classification.

---

##  Dataset

**Source:** [Kaggle – Resume Dataset by Snehaan Bhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

**About the Dataset:**
- Contains 2400+ resumes in both PDF and text format.
- Extracted from [livecareer.com](https://www.livecareer.com/).
- Each resume belongs to one of 24 predefined categories.

**Dataset Columns:**
| Column Name | Description |
|--------------|--------------|
| `ID` | Unique identifier and PDF filename |
| `Resume_str` | Resume text content |
| `Resume_html` | Resume in HTML format (from scraping) |
| `Category` | Job role category label |

**Categories Include:**  
HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

---

##  Methodology

### **1. Data Preprocessing**
- Removed unnecessary symbols, URLs, and extra spaces.
- Lowercased all text.
- Removed stopwords and lemmatized words.

### **2. Feature Extraction**
We experimented with multiple NLP vectorization methods:
| Method | Description | Result |
|--------|-------------|--------|
| **CountVectorizer** | Basic word frequency representation | Moderate accuracy (~67%) |
| **TF–IDF Vectorizer** | Weighted term frequency to reduce common word bias | Improved accuracy (~75%) |
| **BERT Embeddings** | Contextual word representations from pretrained BERT | Best performance (~78%) |

After experimentation, **BERT embeddings** were chosen for the final model due to their ability to capture **semantic meaning** and contextual relationships in text.

### **3. Model Training**
- Split data: 80% training, 20% testing.
- Trained multiple classifiers:
  - Random Forest  
  - Logistic Regression  
  - SVM  
  - Naive Bayes  
  - XGBoost  
  - Gradient Boosting  
  - Passive Aggressive

**Best Performer:**  
 **Logistic Regression + BERT Embeddings**  
This combination achieved the most balanced precision and recall across all categories.

### **4. Deployment**
- Built an interactive **Gradio UI** to allow users to:
  - Type or paste resume text.
  - Upload a PDF resume.
- The app processes the text and predicts the most suitable job category.

---

##  Results

| Model | Vectorization | Accuracy |
|--------|----------------|-----------|
| Random Forest | CountVectorizer | 67% |
| XGBoost | TF-IDF | 75% |
| Logistic Regression | BERT Embeddings | **~78%** |


---

##  Technologies Used

- **Python**
- **Transformers (BERT)** – for text embeddings  
- **Scikit-learn** – for model training and evaluation  
- **Gradio** – for web-based UI  
- **PyPDF2** – for PDF text extraction  
- **Joblib** – for model persistence  

---

