# Detailed Description of the Customer Segmentation Project

## Overview of the Project

The **Customer Segmentation** project is a practical application of machine learning in the fields of business intelligence and marketing analytics. This project focuses on analyzing customer purchasing behavior from real transaction data to group customers into segments with similar characteristics.

The main objectives of the project are to build an automated system that can:

- Analyze patterns in customer transaction data  
- Segment customers into groups that have meaningful business value  
- Provide insights to optimize marketing strategies and customer relationship management  
- Predict the potential value of each customer segment  

## Problem Statement and Business Context

In today’s digital era, retail businesses face a major challenge: **how to truly understand their customers** in order to:

### Real-world challenges:

1. **Increasing marketing costs** – Budgets need to be optimized for the right target audience  
2. **High customer acquisition cost** – Businesses must focus more on retention rather than just acquisition  
3. **Intense competition** – Requires differentiation through personalization  
4. **Rich customer data but underutilized** – Businesses often fail to extract meaningful insights from their data  


### Specific Business Questions:

- **"How can we segment 1,628 customers into meaningful groups?"**
- **"Which customer segments are the most valuable to invest in?"**
- **"What marketing strategies are suitable for each segment?"**

### Proposed Solution:

Use **unsupervised machine learning** to automatically detect hidden patterns in the data and segment customers based on their actual purchasing behavior.


## Introduction to Supervised and Unsupervised Learning

### Supervised Learning

**Definition**: A machine learning approach that uses labeled data to train a model.

**Characteristics**:

- Has a clearly defined target variable (y)
- The model learns from input–output pairs
- Purpose: to predict the output for new, unseen inputs

**Examples**:

- Classification: Predicting whether an email is spam or not
- Regression: Predicting house prices based on area, location, etc.

**General Formulation**: `f(X) = y`

---

### Unsupervised Learning

**Definition**: A machine learning approach that discovers hidden patterns in **unlabeled** data.

**Characteristics**:

- No target variable
- The model automatically uncovers the structure of the data
- Purpose: finding patterns, groups, and associations

**Main Categories**:

1. **Clustering**: Grouping data into clusters (K-means, Hierarchical)
2. **Association Rules**: Finding relationships or co-occurrence patterns (Market Basket Analysis)
3. **Dimensionality Reduction**: Reducing the number of features (PCA, t-SNE)


### Why Choose Unsupervised Learning for This Problem?

**Main Reasons:**

- **No ground truth**: We do not know in advance which segment each customer belongs to  
- **Natural discovery**: We want the data itself to reveal underlying patterns  
- **Flexibility**: Not constrained by predefined segment definitions  
- **Scalability**: Can be applied to any dataset regardless of size or domain  

---

## Approach

### 1. Comprehensive Customer Behavior Analysis

Instead of relying solely on the traditional RFM framework, this project uses a **multi-dimensional analysis** with **16 customer-level features** to capture a more complete picture of purchasing behavior.  
RFM is used only as a **reference method** for visualization and validating segmentation results.

### 2. Data Processing Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Transformation → Clustering → Validation
```

**Step-by-Step Details:**

1. **Data Cleaning**:
   - Remove canceled transactions  
   - Focus on customers from the North American region (United States and Canada)  
   - Handle missing values  

2. **Feature Engineering**:
   - Create 16 comprehensive customer-level features  
   - Aggregate transaction data from multiple perspectives  
   - Use RFM analysis as a reference for visualization  

3. **Data Transformation**:
   - Apply Box-Cox transformation for distribution normalization  
   - Use StandardScaler for feature scaling  

4. **Clustering**:
   - Perform K-means clustering with the optimal number of clusters  
   - Use the Elbow method and Silhouette analysis  

5. **Validation (Simplified for this project)**:
   - Business interpretation of the resulting clusters  


## Data Details

### Data Source

- **Dataset**: Online Retail Data from [ONYXDATA](https://datadna.onyxdata.co.uk/challenges/november-2025-datadna-ecommerce-analytics-challenge/)  
- **Industry**: Software and Applications (SaaS)  
- **Time Period**: April 2024 – October 2025  
- **Geography**: Primarily North America (NA), with some coverage in Europe and globally  

### Raw Data Structure


| Field                | Description                     | Example                                   |
|----------------------|---------------------------------|--------------------------------------------|
| `event_id`           | Transaction ID                  | DDF4B24A8C91                               |
| `event_date`         | Transaction timestamp           | 2025-06-21 10:03:00                        |
| `customer_id`        | Customer ID                     | CUST0003443                                |
| `product_id`         | Product code                    | PROD0065                                   |
| `product_name`       | Product name                    | Microsoft Copilot for Office Monthly       |
| `vendor_name`        | Vendor                          | Microsoft                                  |
| `country_name`       | Country                         | Canada                                     |
| `region`             | Region                          | NA                                         |
| `quantity`           | Quantity                        | 1                                          |
| `unit_price_local`   | Unit price (local currency)     | 19.00                                      |
| `net_revenue_local`  | Net revenue after discount      | 19.95                                      |
| `discount_local`     | Discount amount                 | 0.0                                        |
| `is_refunded`        | Refunded or not                 | False                                      |

### Descriptive Statistics

**Raw Data**:

- **Total Transactions**: 48,000 records  
- **Unique Customers**: 4,000 customers  
- **Unique Products**: 101 products  
- **Countries**: 10 countries  

**After Cleaning (North America only)**:

- **Valid Transactions**: 19,176 records  
- **Customers**: 1,628 customers  
- **Time Period**: 547 days  

### Data Characteristics


**Challenges**:

1. **Negative Quantity**: Canceled or returned transactions  
2. **Extreme Values**: Some transactions have very high values  
3. **Skewed Distribution**: Skewed distribution of customer behavior features  

**Opportunities**:

1. **Rich Transactional Data**: Detailed information on customer spending  
2. **Time Series**: Trends can be analyzed over time  
3. **Product Diversity**: Multiple product categories



## Feature Engineering Approach

### 1. Comprehensive Customer Feature Set

Instead of relying solely on the traditional RFM framework, this project builds a **comprehensive set of 16 customer-level features** to capture all aspects of purchasing behavior.  
RFM is used only as a **reference** for customer visualization.

### Feature Descriptions

The 16 customer-level features capture different aspects of purchasing behavior:

**Basic Metrics:**

- 1. `Discount_Affinity`: Sensitivity to discounts (proportion of transactions with a discount)  
- 2. `AOV`: Average Order Value  
- 3. `Recency`: Number of days since the most recent purchase  
- 4. `Count_Orders`: Total number of orders placed  
- 5. `Sum_Quantity`: Total quantity of products purchased  

**Aggregated by Product:**

- 6. `Count_Products`: Number of unique products purchased  
- 7. `Avg_UnitPricePerProduct`: Average unit price per product  
- 8. `Avg_QuantityPerProduct`: Average quantity per product  
- 9. `Avg_TotalPricePerProduct`: Average spending per product  

**Aggregated by Invoice:**

- 10. `Avg_TotalPricePerInvoice`: Average total value per invoice  
- 11. `Max_Total_Order_Value`: Highest invoice value  
- 12. `Min_Total_Order_Value`: Lowest invoice value  
- 13. `Max_Avg_Order_Value`: Maximum average order value recorded  
- 14. `Min_Avg_Order_Value`: Minimum average order value recorded  

**Aggregated by Order:**

- 15. `Mean_UnitPrice`: Average unit price per line item  
- 16. `Mean_QuantityPerOrder`: Average quantity per order


## Introduction to Box-Cox Transformation

### Why Transformation is Needed?

**Issues with raw customer behavior data**:

1. **Skewed distribution**: The 16 features often have right-skewed distributions due to the nature of business data  
2. **Different scales**: Quantity (units), Price (currency), Count (numbers) have different scales  
3. **Outliers**: High-value customers create extreme values  
4. **Clustering sensitivity**: K-means is sensitive to differences in scale and distribution  

### What is Box-Cox Transformation?

**Definition**: [Box-Cox](https://www.geeksforgeeks.org/data-science/box-cox-transformations/) is a family of power transformations used to normalize distributions.

### Implementation in the Project

**Steps**:

1. **Shift features**: Ensure all values are > 0  
2. **Find optimal λ**: For each feature individually  
3. **Apply transformation**: Transform each feature  
4. **Standardization**: Apply StandardScaler after transformation



## Clustering Using K-means and How It Works

### K-means Algorithm Overview

**Definition**: [K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) is one of the most popular clustering algorithms.  
It partitions n observations into k clusters so that each observation belongs to the cluster with the nearest mean.

---

## Introduction to Notebooks and Source Code

### Source Code Structure

#### 1. clustering_library.py - Core Library

**Class DataCleaner**:


```python
class DataCleaner:
    """Handles data cleaning and basic EDA"""

    def load_data():                  # Load and format data
    def clean_data():                 # Remove invalid records
    def explore_customers():          # Customer-level analysis
    def create_comprehensive_features():  # Generate 16 customer features
    def create_rfm_reference():          # RFM for visualization reference
```

**Class DataVisualizer**:

```python
class DataVisualizer:
    """Visualization and reporting"""

    def plot_missing_data():           # Missing value heatmap
    def plot_sales_trends():           # Time series analysis
    def plot_customer_dist():          # Customer distribution
    def plot_feature_distributions():  # 16 features + RFM distributions
```

**Class FeatureEngineer**:

```python
class FeatureEngineer:
    """Feature engineering and transformations"""

    def create_customer_features():  # Generate 16 comprehensive features
    def transform_features():         # Box-Cox + scaling for all features
    def fit_transform():              # Full pipeline
```

**Class CustomerSegmentAnalyzer**:

```python
class CustomerSegmentAnalyzer:
    """Clustering and analysis"""

    def find_optimal_clusters():  # Elbow + silhouette
    def fit_kmeans():              # Train clustering model
    def analyze_segments():        # Business interpretation
    def plot_clusters():           # Visualization
```

#### 2. Design Patterns

**Object-Oriented Design**:

- Each class has a clear responsibility  
- Encapsulation of methods and attributes  
- Reusability across different datasets

**Pipeline Pattern**:

```python
# Composable Processing Pipeline
cleaner = DataCleaner(data_path)
engineer = FeatureEngineer()
analyzer = CustomerSegmentAnalyzer()

# Pipeline execution
df_clean = cleaner.clean_data()
features = engineer.fit_transform(df_clean)
segments = analyzer.fit_predict(features)
```

### Details of Each Notebook

#### 1. 01_cleaning_and_eda.ipynb

**Objective**: Data cleaning and initial exploration


**Sections**:

1. **Data Loading & Overview**

   - Load 48K transactions
   - Data types and memory usage
   - Missing value analysis

2. **Data Cleaning Process**

   - Remove canceled orders
   - Focus on NA customers only
   - Handle missing CustomerIDs
   - Result: 19.7K valid transactions

3. **Exploratory Data Analysis**

   - Sales trends over time  
   - Customer transaction patterns  
   - Product analysis  
   - Geographic distribution


4. **Key Insights Discovery**
   - 80-20 rule validation
   - Seasonal patterns
   - Customer behavior clusters
   - Data quality assessment

**Outputs**: Clean dataset ready for feature engineering

#### 2. 02_feature_engineering.ipynb

**Objective**: Create customer-level features for clustering

**Sections**:

1. **RFM Features Creation**

   - Aggregate transaction data
   - Calculate R, F, M for each customer
   - Validate business logic

2. **Extended Features**

   - Average basket value
   - Purchase behavior metrics
   - Customer lifecycle features

3. **Distribution Analysis**

   - Feature distributions visualization
   - Skewness and outlier detection
   - Correlation analysis

4. **Data Transformation**
   - Box-Cox transformation cho normality
   - StandardScaler cho equal weights
   - Validation of transformation quality

**Outputs**: Transformed feature matrix ready for clustering

#### 3. 03_modeling.ipynb

**Objective**: Build and evaluate the clustering model


**Sections**:

1. **Optimal Clusters Selection**

   - Elbow method implementation
   - Silhouette analysis
   - Gap statistic (optional)
   - Business constraints consideration

2. **K-means Implementation**

   - Model training with optimal k
   - Multiple initialization runs
   - Convergence monitoring

3. **Cluster Analysis**

   - Cluster centroids interpretation
   - Segment size and characteristics
   - Business meaning of each cluster


4. **Validation & Evaluation**

   - Internal metrics (silhouette, WCSS)
   - Business validation
   - Actionability assessment
   - Stability testing

5. **Results Visualization**
   - 2D/3D cluster plots
   - Segment comparison charts
   - Customer journey mapping

**Outputs**: Final segmentation model and business insights



## Conclusion

### Achievements

#### 1. Technical Achievements

- **Automated Pipeline**: Built an automated pipeline from raw data to final segments  
- **Robust Preprocessing**: High-quality data cleaning and feature engineering  
- **Optimized Clustering**: K-means with k=3,4 optimized based on multiple criteria  

#### 2. Business Impact

- **Customer Insights**: Clear understanding of key customer groups with distinct characteristics  
- **Actionable Segments**: Each segment has specific marketing strategies  
- **Data-Driven Decisions**: Foundation for personalized marketing campaigns  
- **Performance Metrics**: Baseline to measure improvement over time


#### 3. Model Performance

```
Final Clustering Results:
- 4 clusters with balanced sizes  
- Clear business interpretation  
- Stable across multiple runs

```

### Future Directions

#### 1. Model Enhancement

- **Try other algorithms**: DBSCAN, Hierarchical clustering, Gaussian Mixture
- **Feature expansion**: Seasonal features, product categories, geographic
- **Dynamic segmentation**: Time-based evolving segments
- **Ensemble methods**: Combine multiple clustering approaches

#### 2. Business Applications

- **Recommendation System**: Product recommendations for each segment  
- **Price Optimization**: Dynamic pricing based on segments  
- **Churn Prediction**: Supervised learning for at-risk customers  
- **CLV Modeling**: Customer Lifetime Value prediction  
- **Multi-Agent Applications**: Use segment-specific features to simulate customer behavior with AI agents. Discussions between multiple agents can inform more personalized marketing strategies for each segment  

### Final Conclusion

This **Customer Segmentation** project has:

1. **Transformed raw transaction data into actionable business insights**  
2. **Built an automated pipeline applicable to similar datasets**  
3. **Provided a foundation for advanced customer analytics**  
4. **Demonstrated the value of data science in business applications**  

This serves as a clear example of applying machine learning to solve real-world business problems, from data understanding to model deployment and measuring business impact.

