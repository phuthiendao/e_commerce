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

## Summary of results
## Customer Segmentation Workflow (North American SaaS Retail Data)

## 1. Data Filtering
- From the raw SaaS retail dataset, filter and extract data specifically for the **North American** region.

## 2. Exploratory Data Analysis (EDA)
- Analyze the dataset to uncover basic insights and meaningful information.

## 3. Feature Engineering
- Create new, useful features from the original dataset that are suitable for the **K-means clustering algorithm** (customer segmentation).

## 4. Data Transformation
- Apply **Box-Cox transformation** and **standardization** to the engineered features to ensure optimal algorithm performance.

## 5. Determining Optimal Number of Clusters
- Use the following methods to identify the optimal number of clusters:
  - **Elbow Method**
  - **Silhouette Score**
  - **Principal Component Analysis (PCA)**

## 6. Customer Segmentation with K-means
- Apply the **K-means algorithm** to segment customers into **4 main clusters**.
  - **Cluster 0 – High-Potential Customers (20%)**
  - **Cluster 1 – VIP Enterprise (28%)**
  - **Cluster 2 – Economy Tier (21%)**
  - **Cluster 3 – Loyal Regulars (31%)**
  
- Visualize the clusters using:
  - **Radar charts**
  - **SHAP** values to interpret the characteristics and meaning of each cluster.

## 7. Business Strategy Recommendations
- Propose appropriate business strategies tailored to each customer cluster.


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
