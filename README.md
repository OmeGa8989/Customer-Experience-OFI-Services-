# 🧠 Customer Experience Dashboard (Transformer-Enhanced)

An AI-powered **Customer Experience Dashboard** that integrates **Transformer-based NLP models (BERT)** and **classical machine learning** to identify **at-risk customers** and provide **actionable retention strategies**.

The project combines structured (orders, delivery) and unstructured (feedback text) data to predict customer risk levels, visualize customer behavior, and recommend interventions — all within an interactive Streamlit app.

---

## 🚀 Key Highlights

 **Transformer-enhanced text analysis** using SentenceTransformers (BERT-based embeddings)  
 **Hybrid ML model** combining text embeddings + tabular features  
 **Interactive Streamlit Dashboard** with visual analytics  
 **Automated intervention engine** for at-risk customers  
 **Exportable CSV reports** for decision-making  
 **Customizable training pipeline** for new datasets  

---

## 🗂️ Project Structure

```
OFI/
├── app.py                        # Streamlit dashboard (frontend visualization)
├── model_training.py             # Model training and pipeline creation
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation (this file)
├── .gitignore                    # Git ignore file
└── data/                         # Local data folder (excluded from Git)
    ├── orders.csv
    ├── customer_feedback.csv
    ├── delivery_performance.csv
    └── customer_risk_model.joblib
```

> ⚠️ The `data/` folder is **excluded from Git** (for privacy and large file handling).  
You can keep your datasets here locally.

---

## ⚙️ Environment Setup

### 1️⃣ Create a Virtual Environment
```bash
python -m venv .venv
```

### 2️⃣ Activate Virtual Environment
#### 🪟 Windows:
```bash
.venv\Scripts\activate
```

#### 🐧 macOS / Linux:
```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

All dependencies are listed in [`requirements.txt`](./requirements.txt).

```txt
streamlit
pandas
numpy
scikit-learn
joblib
sentence-transformers
transformers
torch
plotly
imbalanced-learn
tqdm
matplotlib
```

Install them all:
```bash
pip install -r requirements.txt
```

---

## 🧩 Model Training

Train your customer risk model locally with:

```bash
python model_training.py
```

### Optional Parameters:
```bash
python model_training.py --use_class_weight    # Handle imbalance via weighted classes
python model_training.py --oversample          # Apply oversampling for minority class
python model_training.py --cv                  # Perform cross-validation evaluation
```

### What Happens:
- Loads and merges data from the `data/` directory  
- Encodes customer feedback text using **Transformer embeddings (MiniLM/BERT)**  
- Combines embeddings with numeric features (Recency, Frequency, Monetary, etc.)  
- Trains a **RandomForestClassifier**  
- Saves model artifact as:  
  ```
  data/customer_risk_model.joblib
  ```

---

## 📊 Run the Streamlit Dashboard

Start the dashboard app:
```bash
streamlit run app.py
```

Then open:
👉 **http://localhost:8501**

---

## 💻 Dashboard Overview

### 🧩 Sections:
1. **Risk Overview** – View total customers segmented by risk levels.  
2. **Recency vs Frequency** – Understand purchasing patterns.  
3. **Monetary Distribution** – Compare spending by risk category.  
4. **Correlation Heatmap** – Analyze feature relationships.  
5. **Top At-Risk Customers** – View suggested interventions.  

### 📈 Visualizations:
| Chart | Purpose |
|--------|----------|
| Risk Distribution (Bar) | Overall risk segmentation |
| Recency vs Frequency (Scatter) | Purchase behavior trends |
| Monetary Box Plot | Spending distribution by risk |
| Correlation Heatmap | Feature correlation visualization |

---

## 💡 Use Cases

- Identify **customers likely to churn**
- Understand **feedback sentiment impact**
- Prioritize **retention campaigns**
- Monitor **delivery performance influence**
- Combine **behavior + sentiment analytics** for smarter marketing

---

## 🧠 Tech Stack

| Component | Technology |
|------------|------------|
| **Frontend** | Streamlit, Plotly |
| **ML Pipeline** | scikit-learn, RandomForest |
| **NLP Model** | SentenceTransformers (MiniLM / BERT) |
| **Visualization** | Plotly, Matplotlib |
| **Data Handling** | pandas, numpy |
| **Language** | Python 3.10+ |

---

## 🧰 Future Enhancements

- SHAP-based explainability (feature importance)
- Real-time API integration (e.g., Google Reviews, Zendesk)
- Streamlit Cloud / AWS EC2 deployment
- Automated alerts for high-risk customer changes
- Role-based access with login support

---

## 🧑‍💻 Author

**Adhyan Aditya**  
📧 *adhyanaditya88@gmail.com*  
🌐 [GitHub Profile](https://github.com/OmeGa8989)

---

## 🔗 Repository Access

The complete project source code is available on GitHub:  
👉 [Customer Experience OFI Services](https://github.com/OmeGa8989/Customer-Experience-OFI-Services-)

This includes:
- Streamlit dashboard (`app.py`)
- Model training pipeline (`model_training.py`)
- Dependencies (`requirements.txt`)
- Documentation (`README.md`)

---

## 🛠️ Commands Summary

```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate         # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train model
python model_training.py

# Run Streamlit dashboard
streamlit run app.py

# GitHub push commands
git init
git add .
git commit -m "Initial commit: Transformer-enhanced Customer Experience Dashboard"
git branch -M main
git remote add origin https://github.com/OmeGa8989/Customer-Experience-OFI-Services-.git
git push -u origin main
```

---

## 🛡️ License

This project is licensed under the **MIT License**.  
Feel free to fork, enhance, and adapt it for academic or professional use.

---

⭐ **If you found this project helpful, please consider giving it a star on GitHub!**
