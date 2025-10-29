# 🧠 Customer Experience Dashboard 
An intelligent **Streamlit dashboard** and **Transformer-based ML pipeline** to identify **at-risk customers** and recommend personalized interventions using behavior, delivery performance, and feedback data.

---

## 🚀 Features
- **Interactive Streamlit Dashboard**
  - Risk segmentation by **High**, **Medium**, and **Low**
  - 4 interactive visualizations using Plotly:
    - 📊 Risk Distribution Bar Chart
    - 🔵 Recency vs Frequency Scatter Plot
    - 💰 Monetary Distribution Box Plot
    - 🔥 Correlation Heatmap
  - Actionable customer intervention suggestions  
  - Downloadable risk predictions as CSV

- **Model Training Script**
  - Transformer-based text embeddings (SentenceTransformers)
  - Tabular + Text features combined into one model
  - Class balancing (via `class_weight` or oversampling)
  - Saved pipeline (`customer_risk_model.joblib`)

---

## 🗂️ Project Structure
OFI/
├── app.py # Streamlit dashboard
├── model_training.py # Model training pipeline
├── requirements.txt # Dependencies
├── README.md # Documentation
├── .gitignore # Ignored files
└── data/ # Local data (excluded from Git)
├── orders.csv
├── customer_feedback.csv
├── delivery_performance.csv
└── customer_risk_model.joblib

2️⃣ Create and Activate Virtual Environment
python -m venv .venv
# Activate:
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Dependencies
pip install -r requirements.txt

🧩 Model Training

Train your model locally:

python model_training.py


Optional parameters:

python model_training.py --use_class_weight
python model_training.py --oversample


After training, the model will be saved to:

data/customer_risk_model.joblib

📊 Run the Dashboard

Launch Streamlit:

streamlit run app.py


Then open the local URL (usually http://localhost:8501/
) in your browser.

Dashboard Views:

Risk Overview (summary of customer risk levels)

Recency vs Frequency (customer behavior scatter)

Monetary Distribution (spending habits by risk)

Correlation Heatmap (feature relationships)

🧠 Tech Stack
Component	Technology Used
Frontend	Streamlit + Plotly
ML Model	Scikit-learn + SentenceTransformers
Data	Pandas, NumPy
Language	Python 3.10+

💡 Example Use Cases
Identify customers likely to churn
Detect risk patterns based on delivery delays and feedback tone
Prioritize retention offers and loyalty rewards
Analyze customer value vs risk trade-offs
