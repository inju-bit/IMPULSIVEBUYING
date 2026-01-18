"""
Behavioral Intention Classification System
Streamlit App for ML Model Training & Prediction
By Inju Khadka - MRes Artificial Intelligence, University of Wolverhampton
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
import io

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.cross_decomposition import PLSRegression
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Page configuration
st.set_page_config(
    page_title="IB Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A6FA5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🧠 Behavioral Intention Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">MRes Artificial Intelligence Research Project - University of Wolverhampton</p>', unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'best_models' not in st.session_state:
    st.session_state.best_models = {}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Navigation")
    
    uploaded_file = st.file_uploader("📁 Upload CSV Data", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(st.session_state.df)} rows")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This application performs:
    - 📊 Exploratory Data Analysis
    - 🔗 Cramér's V Correlation
    - 🤖 ML Model Training
    - 📈 Results Visualization
    """)

# Main content
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 EDA", 
        "🔗 Correlations",
        "🤖 ML Models",
        "📈 Results"
    ])
    
    # ==================== TAB 1: DATA OVERVIEW ====================
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        with col4:
            st.metric("Memory (KB)", f"{df.memory_usage(deep=True).sum()/1024:.1f}")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Unique': df.nunique().values,
            'Missing': df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
    
    # ==================== TAB 2: EDA ====================
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        # Variable Classification
        st.subheader("Variable Classification")
        
        def detect_variable_type(s):
            non_null = s.dropna()
            n_unique = non_null.nunique()
            if n_unique == 0:
                return "Unknown", "Unknown"
            if pd.api.types.is_numeric_dtype(s):
                if np.all(np.mod(non_null, 1) == 0):
                    if n_unique == 2:
                        return "Categorical", "Binary"
                    elif n_unique <= 10:
                        return "Categorical", "Ordinal"
                    else:
                        return "Numeric", "Interval/Ratio"
                return "Numeric", "Interval/Ratio"
            if pd.api.types.is_object_dtype(s):
                if n_unique == 2:
                    return "Categorical", "Binary"
                elif n_unique <= 10:
                    return "Categorical", "Nominal"
                else:
                    return "Categorical", "High-cardinality"
            return "Unknown", "Unknown"
        
        var_types = []
        for col in df.columns:
            basic, detailed = detect_variable_type(df[col])
            var_types.append({"Column": col, "Basic Type": basic, "Detailed Type": detailed})
        
        var_df = pd.DataFrame(var_types)
        st.dataframe(var_df, use_container_width=True)
        
        # Target Variable Analysis
        st.subheader("Target Variable Analysis (IB1, IB2, IB3)")
        
        target_vars = ['IB1', 'IB2', 'IB3']
        available_targets = [t for t in target_vars if t in df.columns]
        
        if available_targets:
            cols = st.columns(len(available_targets))
            for i, var in enumerate(available_targets):
                with cols[i]:
                    st.markdown(f"**{var} Distribution**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    counts = df[var].value_counts().sort_index()
                    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(counts)))
                    ax.bar(counts.index.astype(str), counts.values, color=colors)
                    ax.set_xlabel(var)
                    ax.set_ylabel('Count')
                    ax.set_title(f'{var} Distribution')
                    for j, v in enumerate(counts.values):
                        ax.text(j, v + 0.5, str(v), ha='center', fontsize=9)
                    st.pyplot(fig)
                    plt.close()
        
        # Distribution plots for numeric columns
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select columns to visualize:",
                numeric_cols,
                default=numeric_cols[:min(6, len(numeric_cols))]
            )
            
            if selected_cols:
                n_cols = 3
                n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if len(selected_cols) == 1 else axes
                
                for idx, col in enumerate(selected_cols):
                    ax = axes[idx] if len(selected_cols) > 1 else axes
                    df[col].hist(ax=ax, bins=20, color='steelblue', edgecolor='white')
                    ax.set_title(col)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                
                # Hide empty subplots
                for idx in range(len(selected_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # ==================== TAB 3: CORRELATIONS ====================
    with tab3:
        st.header("🔗 Correlation Analysis")
        
        # Cramér's V Correlation
        st.subheader("Cramér's V Correlation Matrix")
        
        def cramers_v(x, y):
            table = pd.crosstab(x, y)
            chi2 = chi2_contingency(table, correction=False)[0]
            n = table.sum().sum()
            phi2 = chi2 / n
            r, k = table.shape
            phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
            rcorr = r - ((r - 1)**2) / (n - 1)
            kcorr = k - ((k - 1)**2) / (n - 1)
            denom = min((kcorr - 1), (rcorr - 1))
            return np.sqrt(phi2corr / denom) if denom > 0 else 0
        
        # Select categorical columns
        cat_cols = [c for c in df.columns if df[c].nunique() <= 10]
        
        if len(cat_cols) > 1:
            with st.spinner("Computing Cramér's V correlation matrix..."):
                cat_df = df[cat_cols].dropna()
                matrix = pd.DataFrame(
                    np.zeros((len(cat_cols), len(cat_cols))),
                    index=cat_cols, columns=cat_cols
                )
                
                for i in range(len(cat_cols)):
                    for j in range(i, len(cat_cols)):
                        v = cramers_v(cat_df[cat_cols[i]], cat_df[cat_cols[j]])
                        matrix.iloc[i, j] = matrix.iloc[j, i] = v
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    matrix, annot=True, cmap="coolwarm", fmt=".2f",
                    ax=ax, cbar_kws={'label': "Cramér's V"}
                )
                ax.set_title("Cramér's V Correlation Matrix", fontsize=14, pad=15)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # Numeric correlation
        st.subheader("Numeric Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap="RdYlBu_r", fmt=".2f", ax=ax)
            ax.set_title("Pearson Correlation Matrix", fontsize=14, pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # ==================== TAB 4: ML MODELS ====================
    with tab4:
        st.header("🤖 Machine Learning Models")
        
        target_vars = ['IB1', 'IB2', 'IB3']
        available_targets = [t for t in target_vars if t in df.columns]
        
        if not available_targets:
            st.warning("No target variables (IB1, IB2, IB3) found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_target = st.selectbox("Select Target Variable:", available_targets)
            
            with col2:
                selected_models = st.multiselect(
                    "Select Models:",
                    ["Random Forest", "Extra Trees", "Gradient Boosting", 
                     "XGBoost", "LightGBM", "CatBoost"],
                    default=["Random Forest", "XGBoost", "LightGBM"]
                )
            
            col3, col4 = st.columns(2)
            with col3:
                test_size = st.slider("Test Size:", 0.1, 0.4, 0.2, 0.05)
            with col4:
                use_optuna = st.checkbox("Use Optuna Optimization", value=False)
                n_trials = st.number_input("Optuna Trials:", 5, 50, 10, disabled=not use_optuna)
            
            use_oversampling = st.checkbox("Use Random Oversampling (SMOTE)", value=True)
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    # Prepare data
                    df_model = df.copy()
                    
                    # Encode object columns
                    label_encoders = {}
                    for col in df_model.columns:
                        if df_model[col].dtype == "object":
                            le = LabelEncoder()
                            df_model[col] = le.fit_transform(df_model[col].astype(str))
                            label_encoders[col] = le
                    
                    # Convert target to 0-4 scale
                    if df_model[selected_target].min() >= 1:
                        df_model[selected_target] = df_model[selected_target].astype(int) - 1
                    
                    X = df_model.drop(columns=[selected_target])
                    y = df_model[selected_target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                    
                    if use_oversampling:
                        ro = RandomOverSampler(random_state=42)
                        X_train, y_train = ro.fit_resample(X_train, y_train)
                    
                    results = []
                    trained_models = {}
                    
                    # Model classes
                    model_classes = {
                        "Random Forest": RandomForestClassifier,
                        "Extra Trees": ExtraTreesClassifier,
                        "Gradient Boosting": GradientBoostingClassifier,
                        "XGBoost": xgb.XGBClassifier,
                        "LightGBM": lgb.LGBMClassifier,
                        "CatBoost": CatBoostClassifier
                    }
                    
                    # Default parameters
                    default_params = {
                        "Random Forest": {"n_estimators": 200, "max_depth": 10, "random_state": 42},
                        "Extra Trees": {"n_estimators": 200, "max_depth": 10, "random_state": 42},
                        "Gradient Boosting": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
                        "XGBoost": {"n_estimators": 200, "max_depth": 6, "eval_metric": "mlogloss", 
                                   "verbosity": 0, "random_state": 42},
                        "LightGBM": {"n_estimators": 200, "max_depth": 10, "verbose": -1, "random_state": 42},
                        "CatBoost": {"iterations": 200, "depth": 6, "verbose": False, "random_state": 42}
                    }
                    
                    progress_bar = st.progress(0)
                    
                    for idx, model_name in enumerate(selected_models):
                        st.write(f"Training {model_name}...")
                        
                        params = default_params.get(model_name, {})
                        model_class = model_classes[model_name]
                        
                        if use_optuna and model_name in ["Random Forest", "XGBoost", "LightGBM"]:
                            # Simple Optuna optimization
                            def objective(trial):
                                if model_name == "Random Forest":
                                    p = {
                                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                                        "max_depth": trial.suggest_int("max_depth", 4, 15),
                                        "random_state": 42
                                    }
                                    m = RandomForestClassifier(**p)
                                elif model_name == "XGBoost":
                                    p = {
                                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                        "eval_metric": "mlogloss",
                                        "verbosity": 0,
                                        "random_state": 42
                                    }
                                    m = xgb.XGBClassifier(**p)
                                else:  # LightGBM
                                    p = {
                                        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                                        "verbose": -1,
                                        "random_state": 42
                                    }
                                    m = lgb.LGBMClassifier(**p)
                                
                                m.fit(X_train, y_train)
                                preds = m.predict(X_test)
                                return f1_score(y_test, preds, average='weighted')
                            
                            study = optuna.create_study(direction="maximize")
                            study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
                            params.update(study.best_params)
                        
                        # Train model
                        model = model_class(**params)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        
                        # Calculate metrics
                        acc = accuracy_score(y_test, preds)
                        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                        
                        results.append({
                            "Target": selected_target,
                            "Model": model_name,
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1": f1
                        })
                        
                        trained_models[model_name] = {
                            'model': model,
                            'predictions': preds,
                            'y_test': y_test,
                            'X_test': X_test
                        }
                        
                        progress_bar.progress((idx + 1) / len(selected_models))
                    
                    st.session_state.results = pd.DataFrame(results)
                    st.session_state.best_models[selected_target] = trained_models
                    st.session_state.y_test = y_test
                    st.session_state.X_test = X_test
                    
                    st.success("✅ Training complete!")
                    
                    # Display results
                    st.subheader("📊 Training Results")
                    st.dataframe(
                        st.session_state.results.style.highlight_max(
                            subset=['Accuracy', 'F1'], color='lightgreen'
                        ),
                        use_container_width=True
                    )
    
    # ==================== TAB 5: RESULTS ====================
    with tab5:
        st.header("📈 Results & Visualizations")
        
        if st.session_state.results is not None:
            results_df = st.session_state.results
            
            # Metrics comparison
            st.subheader("Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
                bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors)
                ax.set_xlabel('Accuracy')
                ax.set_title('Model Accuracy Comparison')
                ax.set_xlim(0, 1)
                for bar, val in zip(bars, results_df['Accuracy']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{val:.4f}', va='center', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # F1 Score comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(results_df['Model'], results_df['F1'], color=colors)
                ax.set_xlabel('F1 Score')
                ax.set_title('Model F1 Score Comparison')
                ax.set_xlim(0, 1)
                for bar, val in zip(bars, results_df['F1']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{val:.4f}', va='center', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Confusion Matrices
            st.subheader("Confusion Matrices")
            
            if st.session_state.best_models:
                target = list(st.session_state.best_models.keys())[0]
                models = st.session_state.best_models[target]
                
                n_models = len(models)
                n_cols = min(3, n_models)
                n_rows = (n_models + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
                if n_models == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for idx, (model_name, model_data) in enumerate(models.items()):
                    y_test = model_data['y_test']
                    preds = model_data['predictions']
                    labels = sorted(y_test.unique())
                    
                    cm = confusion_matrix(y_test, preds, labels=labels)
                    
                    sns.heatmap(
                        cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        ax=axes[idx]
                    )
                    axes[idx].set_title(f'{model_name}')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')
                
                # Hide empty subplots
                for idx in range(n_models, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # F1 Loss Chart
            st.subheader("Loss Analysis (1 - F1)")
            results_df['F1_Loss'] = 1 - results_df['F1']
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_df = results_df.sort_values('F1_Loss')
            colors = plt.cm.rocket(np.linspace(0.2, 0.8, len(sorted_df)))
            bars = ax.bar(sorted_df['Model'], sorted_df['F1_Loss'], color=colors)
            ax.set_ylabel('Loss (1 - F1)')
            ax.set_title('F1 Loss by Model (Lower is Better)')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            for bar, val in zip(bars, sorted_df['F1_Loss']):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                       f'{val:.3f}', ha='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Download results
            st.subheader("📥 Download Results")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Please train models in the 'ML Models' tab first.")

else:
    # Welcome screen
    st.markdown("""
    ## Welcome! 👋
    
    This application helps you analyze behavioral intention data using machine learning.
    
    ### Features:
    - 📊 **Data Overview**: Explore your dataset structure and statistics
    - 🔍 **EDA**: Visualize distributions and variable types  
    - 🔗 **Correlations**: Cramér's V and Pearson correlation matrices
    - 🤖 **ML Models**: Train multiple classifiers with optional Optuna optimization
    - 📈 **Results**: Compare model performance with visualizations
    
    ### Getting Started:
    1. Upload your CSV file using the sidebar
    2. The file should contain columns: PE1-PE4, SE1-SE3, TP1-TP3, HB1-HB3, UB1-UB3, IB1-IB3
    3. Navigate through the tabs to analyze your data
    
    ### Expected Data Format:
    The system expects a research dataset with:
    - **Predictor variables**: PE (1-4), SE (1-3), TP (1-3), HB (1-3), UB (1-3)
    - **Target variables**: IB1, IB2, IB3 (behavioral intentions on 1-5 scale)
    """)
    
    # Sample data generation option
    if st.button("📝 Generate Sample Data"):
        np.random.seed(42)
        n_samples = 200
        
        sample_data = {
            'PE1': np.random.randint(1, 6, n_samples),
            'PE2': np.random.randint(1, 6, n_samples),
            'PE3': np.random.randint(1, 6, n_samples),
            'PE4': np.random.randint(1, 6, n_samples),
            'SE1': np.random.randint(1, 6, n_samples),
            'SE2': np.random.randint(1, 6, n_samples),
            'SE3': np.random.randint(1, 6, n_samples),
            'TP1': np.random.randint(1, 6, n_samples),
            'TP2': np.random.randint(1, 6, n_samples),
            'TP3': np.random.randint(1, 6, n_samples),
            'HB1': np.random.randint(1, 6, n_samples),
            'HB2': np.random.randint(1, 6, n_samples),
            'HB3': np.random.randint(1, 6, n_samples),
            'UB1': np.random.randint(1, 6, n_samples),
            'UB2': np.random.randint(1, 6, n_samples),
            'UB3': np.random.randint(1, 6, n_samples),
            'IB1': np.random.randint(1, 6, n_samples),
            'IB2': np.random.randint(1, 6, n_samples),
            'IB3': np.random.randint(1, 6, n_samples),
        }
        
        st.session_state.df = pd.DataFrame(sample_data)
        st.success("✅ Sample data generated! Refresh to see the analysis tabs.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Developed by Inju Khadka | MRes Artificial Intelligence | University of Wolverhampton
    </div>
    """,
    unsafe_allow_html=True
)
