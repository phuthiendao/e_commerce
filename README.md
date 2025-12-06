# Customer Segmentation Project Using K-means Clustering Algorithm to Analyze Shopping Behavior and Group Customers

## Objective

Analyze customer transaction data to:

- Understand customer shopping behavior
- Segment customers into groups with similar characteristics
- Develop targeted marketing strategies for each group

## Data

- **Source**: Online retail company transaction data from the UK (2010-2011)
- **Size**: 541,909 transactions from 4,372 customers
- **Features**: Unique gift and household product transactions

## Project Structure

```
├── data/
│   ├── raw/                    # Dữ liệu thô
│   └── processed/              # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb     # Làm sạch dữ liệu và EDA
│   ├── 02_feature_engineering.ipynb  # Xây dựng features
│   └── 03_modeling.ipynb             # Mô hình clustering
├── src/
│   └── clustering_library.py         # Thư viện chính
├── docs/
│   └── project_description.md        # Mô tả chi tiết dự án
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
