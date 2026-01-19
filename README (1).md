# Impulsive Buying Behavior Analysis 🛒

Hi! I'm **Inju Khadka**, an MRes Artificial Intelligence student at the University of Wolverhampton. This is my research project on analyzing impulsive buying behavior using machine learning and PLS-SEM.

## What is this project about?

For my research, I'm studying what factors influence people's impulsive buying decisions. I used collected survey data measuring different psychological constructs and built ML models to predict buying intentions.

## The Variables

My survey measured these factors (each rated 1-5):
 
| Variable | What it measures |
|----------|------------------|
| **PE** (PE1-PE4) | Physical Environment - store layout, atmosphere, visual merchandising |
| **SE** (SE1-SE3) | Social Environment - influence of salespeople, other shoppers |
| **TP** (TP1-TP3) | Time Perspective - time pressure, shopping duration |
| **HB** (HB1-HB3) | Hedonic Browsing - enjoyment, fun, pleasure from browsing |
| **UB** (UB1-UB5) | Utilitarian Browsing - goal-directed, efficient shopping |
| **IB** (IB1-IB3) | Impulse Buying - the target variable (unplanned purchases) |

## What does the app do?

I built this Streamlit app to make my analysis interactive. It includes:

1. **Data Overview** - Basic stats about the dataset
2. **EDA** - Charts showing how responses are distributed
3. **Correlations** - Cramér's V heatmaps to see relationships
4. **ML Models** - Train 6 different classifiers:
   - Random Forest
   - Extra Trees
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost
5. **Results** - Compare accuracy, F1 scores, confusion matrices
6. **PLS-SEM** - Structural equation modeling to test my hypotheses

## How to run it

### Option 1: Streamlit Cloud (easiest)

The app is deployed at: (https://impulsivebuying-j7tgappvxo9fufbzt2aw7ds.streamlit.app/)

Just upload your CSV and start analyzing!

### Option 2: Run locally

```bash
# Clone the repo
git clone https://github.com/inju-bit/IMPULSIVEBUYING.git
cd IMPULSIVEBUYING

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Data Format

Your CSV should look like this:

| PE1 | PE2 | PE3 | PE4 | SE1 | SE2 | SE3 | TP1 | TP2 | TP3 | HB1 | HB2 | HB3 | UB1 | UB2 | UB3 | UB4 | UB5 | IB1 | IB2 | IB3 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 4 | 5 | 3 | 4 | 5 | 4 | 4 | 3 | 4 | 3 | 4 | 5 | 4 | 3 | 4 | 3 | 4 | 5 | 3 | 4 | 3 |

All values should be 1-5 (Likert scale).

## Screenshots

*(coming soon)*

## Results Summary

From my analysis:
- XGBoost and Random Forest generally performed best
- The PLS-SEM showed that Hedonic Benefit (HB) has the strongest influence on impulsive buying
- Trust Perception (TP) also plays a significant role

## Tech Stack

- Python 3.10+
- Streamlit (for the web app)
- Scikit-learn, XGBoost, LightGBM, CatBoost (for ML)
- Matplotlib, Seaborn (for visualization)
- Pandas, NumPy (for data processing)

## Files

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contact

If you have questions about my research or the app:

- **Name:** Inju Khadka
- **Program:** MRes Artificial Intelligence
- **University:** University of Wolverhampton
- **Year:** 2025

## Acknowledgments

Thanks to my supervisor and the University of Wolverhampton for supporting this research.

---

*This project is part of my MRes dissertation research.*

University of Wolverhampton
