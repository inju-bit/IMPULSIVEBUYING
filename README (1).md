# 🧠 Behavioral Intention Classification System

**MRes Artificial Intelligence Research Project**  
**University of Wolverhampton**  
**By Inju Khadka**

---

## 📋 Overview

This Streamlit application provides a complete machine learning pipeline for analyzing behavioral intention data. It supports:

- 📊 Exploratory Data Analysis (EDA)
- 🔗 Cramér's V Correlation Analysis  
- 🤖 Multiple ML Model Training (Random Forest, XGBoost, LightGBM, CatBoost, etc.)
- 🔧 Optuna Hyperparameter Optimization
- 📈 Results Visualization & Comparison
- ⚖️ Class Imbalance Handling (Random Oversampling)

---

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

### Option 2: Local Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:**
   - Navigate to `http://localhost:8501`

### Option 3: Docker Deployment

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t ib-classifier .
   docker run -p 8501:8501 ib-classifier
   ```

### Option 4: Heroku Deployment

1. **Create `Procfile`:**
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. **Create `setup.sh`:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]\nheadless = true\nport = $PORT\nenableCORS = false\n" > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

## 📁 Project Structure

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

---

## 📊 Expected Data Format

The application expects a CSV file with the following columns:

| Category | Columns | Description |
|----------|---------|-------------|
| **PE** | PE1, PE2, PE3, PE4 | Performance Expectancy (1-5 scale) |
| **SE** | SE1, SE2, SE3 | Self-Efficacy (1-5 scale) |
| **TP** | TP1, TP2, TP3 | Trust Perception (1-5 scale) |
| **HB** | HB1, HB2, HB3 | Hedonic Benefit (1-5 scale) |
| **UB** | UB1, UB2, UB3 | Utilitarian Benefit (1-5 scale) |
| **IB** | IB1, IB2, IB3 | Behavioral Intention - **Target Variables** (1-5 scale) |

---

## 🤖 Supported Models

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of decision trees with bagging |
| Extra Trees | Extremely randomized trees |
| Gradient Boosting | Sequential boosting with gradient descent |
| XGBoost | Optimized gradient boosting |
| LightGBM | Fast gradient boosting with histogram-based learning |
| CatBoost | Gradient boosting with categorical feature support |

---

## ⚙️ Features

### Data Analysis
- Automatic variable type detection
- Missing value analysis
- Distribution visualizations

### Correlation Analysis
- Cramér's V for categorical variables
- Pearson correlation for numeric variables
- Interactive heatmaps

### Model Training
- Multiple model selection
- Train/test split configuration
- Random Oversampling for imbalanced data
- Optional Optuna hyperparameter optimization

### Results
- Accuracy, Precision, Recall, F1 metrics
- Confusion matrices for all models
- Loss analysis (1 - F1)
- CSV export functionality

---

## 📝 Usage

1. **Upload Data**: Use the sidebar to upload your CSV file
2. **Explore**: Navigate through tabs to view data analysis
3. **Train**: Select models and configure training parameters
4. **Evaluate**: View results, confusion matrices, and download reports

---

## 🔧 Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload file size limits
- Server settings

---

## 📄 License

This project is part of academic research at the University of Wolverhampton.

---

## 📧 Contact

**Inju Khadka**  
MRes Artificial Intelligence  
University of Wolverhampton
