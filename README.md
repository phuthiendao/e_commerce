# Customer Segmentation Project Using K-means Clustering Algorithm to Analyze Shopping Behavior and Group Customers

## Objective

Analyze customer transaction data to:

- Understand customer shopping behavior
- Segment customers into groups with similar characteristics
- Develop targeted marketing strategies for each group

## Data

- **Source**: Online retail company transaction data from a North American (NA) retail technology company.
- **Size**: 19,579 transactions from 1,628 North American customers
- **Features**:SaaS application transactions

  
## Project Structure

```
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb     # Data cleaning and EDA
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   └── 03_modeling.ipynb             # Clustering model
├── sql_scripts/
|
├── src/
│   └── clustering_library.py         # Main clustering library
├── docs/
│   └── project_description.md        # Detailed project description
└── requirements.txt                  # Dependencies
```

## Quick Start

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run notebooks in order:**
   - `01_cleaning_and_eda.ipynb` - Clean and explore data
   - `02_feature_engineering.ipynb` - Create features
   - `03_modeling.ipynb` - Build the clustering model

## Technologies Used

- **Python**
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Visualization
- **NumPy** - Numerical computations

## Documentation

Details of the methodology and theory are described in `docs/project_description.md`
