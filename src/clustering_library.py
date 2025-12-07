# -*- coding: utf-8 -*-

"""
Customer Segmentation Library

This library contains classes for data cleaning, feature engineering, and clustering
analysis for customer segmentation.
"""

from datetime import datetime
import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import math
import shap
from scipy.stats import boxcox
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler



class DataCleaner:
    """
    A class for cleaning, preprocessing, and calculating RFM metrics
    for retail transaction data.
    """

    def __init__(self, data_path):
        """
        Initialize the DataCleaner with data path.

        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_na = None
        self.rfm_data = None
        
        
        
    def load_data(self):
        """
        Load and display basic information about the dataset.

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            event_id="object",
            event_date="object",  # will convert to datetime with parse_dates
            customer_id="object",
            product_id="object",
            country_name="object",
            region="object",  
            payment_method="object",
            quantity="int64",
            unit_price_local="float64",
            net_revenue_local="float64",
            is_refunded="bool",
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["event_date"],
            dtype=dtype,
        )

        self.df['event_date'] = pd.to_datetime(self.df['event_date'], errors='coerce')
        self.df['region'] = self.df['region'].fillna('NA')

        print(f"Loaded data shape: {self.df.shape}")
        print(f"Total records: {len(self.df):,}")
        return self.df



    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing on North American (NA) customers.

        Returns:
            pd.DataFrame: Cleaned North American (NA) dataset
        """

        # Remove the canceled purchase events
        self.df = self.df[~self.df["is_refunded"]]
        
        # Focus on North American (NA) customers
        self.df_na = self.df[self.df["region"] == "NA"]
        self.df_na.dropna(subset=["customer_id"], inplace=True)
        self.df_na =  self.df_na[( self.df_na["quantity"] > 0) & ( self.df_na["unit_price_local"] > 0)]
        return  self.df_na



    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        self.df_na["DayOfWeek"] = self.df_na["event_date"].dt.dayofweek
        self.df_na["HourOfDay"] = self.df_na["event_date"].dt.hour



    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics.

        Returns:
            pd.DataFrame: RFM data for each customer
        """
        snapshot_date = self.df_na["event_date"].max() + pd.Timedelta(days=1)
        self.rfm_data = self.df_na.groupby("customer_id").agg(
            {
                "event_date": lambda x: (snapshot_date - x.max()).days,  # Recency
                "event_id": lambda x: x.nunique(),                       # Frequency
                "net_revenue_local": lambda x: x.sum(),                 # Monetary
            }
        )
        self.rfm_data.columns = ["Recency", "Frequency", "Monetary"]
        return self.rfm_data



    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.

        Args:
            output_dir (str): Output directory path
        """
        
        os.makedirs(output_dir, exist_ok=True)
        self.df_na.to_csv(f"{output_dir}/cleaned_na_data.csv", index=False)
        print(f"Saved cleaned data: {output_dir}/cleaned_na_data.csv")



class FeatureEngineer:
    """
    A class for creating customer-level features from transaction data.

    This class aggregates transaction-level data into customer-level features
    for clustering analysis.
    """

    def __init__(self, data_path):
        """
        Initialize the FeatureEngineer with data path.
        Args:
            data_path (str): Path to the cleaned data file
        """
        self.data_path = data_path
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.customer_features_scaled = None
        
        # Define customer feature names
        self.feature_customer = [            
        "Discount_Affinity",
        "AOV",  # Average Order Value
        "Recency",
        
        "Count_Orders",
        "Count_Products",
        "Sum_Quantity",
        "Mean_UnitPrice",
        "Mean_QuantityPerOrder",
        
        "Avg_UnitPricePerProduct", 
        "Avg_QuantityPerProduct", 
        "Avg_TotalPricePerProduct", 
        "Avg_TotalPricePerInvoice",
        
        "Max_Avg_Order_Value", 
        "Min_Avg_Order_Value", 
        "Max_Total_Order_Value",
        "Min_Total_Order_Value"

        ]
        self.feature_customer2 = ["customer_id"] + self.feature_customer



    def load_data(self):
        """
        Load cleaned data and prepare for feature engineering.

        Returns:
            pd.DataFrame: Loaded cleaned data
        """
        
        self.df = pd.read_csv(self.data_path)
        self.df["event_date"] = pd.to_datetime(self.df["event_date"], errors='coerce')

        print(f"Data size: {self.df.shape}")
        return self.df



    # CREATE CUSTOMER FEATURES
    def create_customer_features(self):
        """
        Create customer-level aggregated features.

        Returns:
            pd.DataFrame: Customer features dataframe
        """
        num_customers = self.df["customer_id"].nunique()
        
        self.customer_features = pd.DataFrame(
            data=np.zeros((num_customers, len(self.feature_customer2)), dtype=float),
            columns=self.feature_customer2,
        )
        
        self.customer_features["customer_id"] = self.customer_features["customer_id"].astype("object")
        
        print(f"Calculating features for each customer...")

        for i, (customer_id, value) in enumerate(self.df.groupby("customer_id")):
            # Customer ID
            self.customer_features.iat[i, 0] = customer_id

            # Discount Affinity
            self.customer_features.iat[i, 1] = (value["discount_local"] > 0).mean()
            
            # Average Order Value (AOV)
            self.customer_features.iat[i, 2] = value.groupby("event_id")["net_revenue_local"].sum().mean()
            
            # Recency
            last_purchase_date = value["event_date"].max()
            latest_date_in_data = self.df["event_date"].max() + pd.Timedelta(days=1)
            recency = (latest_date_in_data - last_purchase_date).days
            self.customer_features.iat[i, 3] = recency

            # Count Orders and Products
            self.customer_features.iat[i, 4] = value["event_id"].nunique()
            self.customer_features.iat[i, 5] = value["product_id"].nunique()

            # Sum of Quantities
            self.customer_features.iat[i, 6] = value["quantity"].sum()

            # Mean Unit Price
            self.customer_features.iat[i, 7] = value["unit_price_local"].mean()

            # Mean Quantity per Order
            self.customer_features.iat[i, 8] = value.groupby("event_id")["quantity"].sum().mean()

            # Mean Unit Price per Product
            self.customer_features.iat[i, 9] = value.groupby("product_id")["unit_price_local"].mean().mean()

            # Mean Quantity per Product
            self.customer_features.iat[i, 10] = value.groupby("product_id")["quantity"].sum().mean()

            # Mean Net Revenue per Product
            self.customer_features.iat[i, 11] = value.groupby("product_id")["net_revenue_local"].sum().mean()

            # Mean Net Revenue per Event
            self.customer_features.iat[i, 12] = value.groupby("event_id")["net_revenue_local"].sum().mean()

            # Calculate new features

            # Max Average Order Value
            max_avg_order_value = value.groupby("event_id")["net_revenue_local"].sum().mean()
            self.customer_features.iat[i, 13] = max_avg_order_value

            # Min Average Order Value
            min_avg_order_value = value.groupby("event_id")["net_revenue_local"].sum().mean()
            self.customer_features.iat[i, 14] = min_avg_order_value

            # Max Total Order Value
            max_total_order_value = value.groupby("event_id")["net_revenue_local"].sum().max()
            self.customer_features.iat[i, 15] = max_total_order_value

            # Min Total Order Value
            min_total_order_value = value.groupby("event_id")["net_revenue_local"].sum().min()
            self.customer_features.iat[i, 16] = min_total_order_value


            if (i + 1) % 500 == 0:
                print(f"Processed {i+1}/{num_customers} customers...")

        print("Feature calculation completed!")
        return self.customer_features




    # TRANSFORM FEATURES
    def transform_features(self):
        """
        Apply Box-Cox transformation to normalize feature distributions.

        Returns:
            pd.DataFrame: Transformed features
        """
        # # Set customer_id as index
        customer_features_indexed = self.customer_features.set_index("customer_id")
        
        # Apply Box-Cox transformation
        feature_values = customer_features_indexed.values + 1  # Plus 1 for Box-Cox (avoid negative values)
        
        self.customer_features_transformed = customer_features_indexed.copy()

        print("Applying Box-Cox transformation")
        for i, feature in enumerate(self.feature_customer):
            transformed, lambda_param = boxcox(feature_values[:, i])
            self.customer_features_transformed.iloc[:, i] = transformed
        print("Box-Cox transformation completed!")
        return self.customer_features_transformed



    # SCALE FEATURES
    """
        Apply standardization to features.

        Returns:
            pd.DataFrame: Scaled features
    """
    def scale_features(self):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features_transformed)

        self.customer_features_scaled = pd.DataFrame(
            features_scaled,
            columns=self.feature_customer,
            index=self.customer_features_transformed.index,
        )
        print("Feature scaling completed!")
        return self.customer_features_scaled
    
    
    
    # PLOT_FEATURES_BOXPLOTS
    def plot_features_boxplots(self, transformed=False, save_path=None):
        """
        Plot box plots for all features.

        Args:
            transformed (bool): True to plot Box-Cox transformed features, False for original features
            save_path (str): Path to save the image (optional)
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("customer_id")
                title = "Box Plots of Original Features (Before Box-Cox Transformation)"
            else:
                print(
                    "Error: No feature data available. Please run create_customer_features() first."
                )
                return
        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.boxplot(data.iloc[:, i] if transformed else data[feature])
                plt.title(feature, fontsize=10)
                plt.xticks([])

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=1.1)

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"The chart has been saved to: {save_path}")
                
            plt.show()
            
            
    
    # PLOT_FEATURES_HISTOGRAMS     
    def plot_features_histograms(self, transformed=False, save_path=None):
        """
        Plot histograms for all features.
        Args:
            transformed (bool): True to plot Box-Cox transformed features, False for original features
            save_path (str): Path to save the image (optional)
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
            title = "Histograms of Features After Box-Cox Transformation"
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("customer_id")
                title = "Histograms of Original Features (Before Box-Cox Transformation)"
            else:
                print(
                    "Error: No feature data available. Please run create_customer_features() first."
                )
                return

        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.hist(
                    data.iloc[:, i] if transformed else data[feature],
                    bins=30,
                    alpha=0.9,
                )
                plt.title(feature, fontsize=10)
                plt.ylabel("Frequency", fontsize=8)

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Chart saved: {save_path}")

        plt.show()

            
            
    
    # SAVE FEATURES
    def save_features(self, output_dir="../data/processed"):
        """
        Save all processed features.

        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save original features
        customer_features_indexed = self.customer_features.set_index("customer_id")
        customer_features_indexed.to_csv(f"{output_dir}/customer_features.csv")

        # Save transformed features
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )

        # Save scaled features
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )

        print(f"All features have been saved to: {output_dir}")

        
        
class ClusterAnalyzer:
    """
    A class for performing clustering analysis and visualization.

    This class handles PCA, optimal cluster determination, K-means clustering,
    and cluster visualization and interpretation.
    """

    # English feature names mapping
    FEATURE_NAMES_EN = {
        "Discount_Affinity": "Discount Affinity",
        "AOV": "Average Order Value (AOV)",

        "Count_Orders": "Purchase Frequency",
        "Count_Products": "Product Diversity",
        "Sum_Quantity": "Purchase Volume",

        "Mean_UnitPrice": "Average Unit Price",
        "Mean_QuantityPerOrder": "Average Quantity per Order",

        "Avg_UnitPricePerProduct": "Average Unit Price per Product",
        "Avg_QuantityPerProduct": "Average Quantity per Product",
        "Avg_TotalPricePerProduct": "Average Total Spend per Product",
        "Avg_TotalPricePerInvoice": "Average Total Spend per Invoice",

        "Max_Avg_Order_Value": "Max Average Order Value",
        "Min_Avg_Order_Value": "Min Average Order Value",
        "Max_Total_Order_Value": "Max Total Order Value",
        "Min_Total_Order_Value": "Min Total Order Value",

        "Recency": "Recency"
    }




    def __init__(self, scaled_features_path, original_features_path):
        """
        Initialize the ClusterAnalyzer with feature data paths.

        Args:
            scaled_features_path (str): Path to scaled features file
            original_features_path (str): Path to original features file
        """
        self.scaled_features_path = scaled_features_path
        self.original_features_path = original_features_path
        self.df_scaled = None
        self.df_original = None
        self.df_pca = None
        self.pca = None
        self.optimal_clusters = {}
        self.cluster_results = {}
        self.surrogate_models = {}
        self.shap_results = {}



    def load_data(self):
        """
        Load scaled and original features data.

        Returns:
            tuple: (scaled_features_df, original_features_df)
        """
        self.df_scaled = pd.read_csv(self.scaled_features_path, index_col=0)
        self.df_original = pd.read_csv(self.original_features_path, index_col=0)

        print(f"Number of Customers: {self.df_scaled.shape[0]}")
        print(f"Number of Features: {self.df_scaled.shape[1]}")

        return self.df_scaled, self.df_original



    # APPLY PCA
    def apply_pca(self, n_components=None):
        """
        Apply Principal Component Analysis.
        Args:
            n_components (int): Number of components to keep
        Returns:
            pd.DataFrame: PCA-transformed data
        """
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(self.df_scaled)

        pca_columns = [f"PC{i+1}" for i in range(pca_features.shape[1])]
        self.df_pca = pd.DataFrame(
            pca_features, columns=pca_columns, index=self.df_scaled.index
        )
        print(f"PCA shape: {self.df_pca.shape}")
        return self.df_pca
    


    # PLOT PCA VARIANCE
    def plot_pca_variance(self):
        """
        Plot explained variance ratio from PCA.
        """
        plt.figure(figsize=(12, 6))

        plt.bar(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            self.pca.explained_variance_ratio_,
            alpha=0.7,
            label="Individual variance",
        )
        plt.step(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            np.cumsum(self.pca.explained_variance_ratio_),
            where="mid",
            label="Cumulative variance",
            color="red",
            linewidth=2,
        )

        plt.axhline(y=0.8, color="green", linestyle="--", label="80% variance")
        plt.axhline(y=0.9, color="orange", linestyle="--", label="90% variance")

        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Analysis - Explained Variance")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nCumulative variance:")
        for i in range(min(5, len(self.pca.explained_variance_ratio_))):
            cumsum = np.sum(self.pca.explained_variance_ratio_[: i + 1])
            print(f"PC1-PC{i+1}: {cumsum:.2%}")


    # FIND OPTIMAL CLUSTERS
    def find_optimal_clusters(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using multiple methods.
        Args:
            k_range (range): Range of k values to test

        Returns:
            dict: Results from different methods
        """
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_scaled, labels))

        self.optimal_clusters = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "best_k_silhouette": list(k_range)[np.argmax(silhouette_scores)],
        }

        return self.optimal_clusters
    
    
    
    # PLOT OPTIMAL CLUSTERS
    def plot_optimal_clusters(self):
        """
        Plot Elbow method and Silhouette scores for cluster selection.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Elbow Method
        axes[0].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["inertias"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="blue",
        )
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True, alpha=0.3)

        # Silhouette Score
        axes[1].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["silhouette_scores"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="green",
        )
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score Method")
        axes[1].grid(True, alpha=0.3)

        best_k = self.optimal_clusters["best_k_silhouette"]
        best_score = max(self.optimal_clusters["silhouette_scores"])
        axes[1].scatter(best_k, best_score, s=200, c="red", alpha=0.5, zorder=5)
        axes[1].annotate(
            f"The best k={best_k}",
            xy=(best_k, best_score),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        plt.tight_layout()
        plt.show()

        print(f"Recommended Silhouette Score: k={best_k} (score = {best_score:.3f})")


    
    # APPLY K-MEANS
    def apply_kmeans(self, k_values=[3, 4]):
        """
        Apply K-means clustering with different k values.
        
        Args:
            k_values (list): List of k values to apply
            
        Returns:
            dict: Clustering results for each k
        """
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.df_scaled)

            # Add clusters to dataframes
            cluster_col = f"Cluster_{k}"
            self.df_scaled[cluster_col] = clusters
            self.df_pca[cluster_col] = clusters
            self.df_original[cluster_col] = clusters

            self.cluster_results[k] = {
                "labels": clusters,
                "sizes": pd.Series(clusters).value_counts().sort_index(),
                "means": self.df_original.groupby(cluster_col).mean(),
            }
            print(f"Size of clusters (k={k}):")
            print(self.cluster_results[k]["sizes"])

        return self.cluster_results
    
    
    
    # PLOT CLUSTERS PCA 2D
    def plot_clusters_pca(self, k_values=[3, 4]):
        """
        Visualize clusters in PCA space.

        Args:
            k_values (list): List of k values to visualize
        """
        fig, axes = plt.subplots(1, len(k_values), figsize=(16, 6))
        if len(k_values) == 1:
            axes = [axes]

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            scatter = axes[i].scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_title(f"K-Means (k={k})")
            plt.colorbar(scatter, ax=axes[i], label="Cluster")

        plt.tight_layout()
        plt.show()



    # PLOT CLUSTERS PCA 3D
    def plot_clusters_pca_3d(self, k_values=[3, 4]):
        """
        Visualize clusters in 3D PCA space.
        Args:
            k_values (list): List of k values to visualize
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 6))

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            ax = fig.add_subplot(1, len(k_values), i + 1, projection="3d")

            scatter = ax.scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                self.df_pca["PC3"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"K-Means 3D (k={k})")

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5)

        plt.tight_layout()
        plt.show()



    # CREATE INDIVIDUAL RADAR PLOTS
    def create_individual_radar_plots(self, k, cluster_names=None): 
        """
        Create individual radar plots for each cluster.

        Args:
            k (int): Number of clusters
            cluster_names (list): Custom names for clusters
        """
        cluster_means = self.cluster_results[k]["means"]

        # Select important features for segmentation
        important_features = {
            "Sum_Quantity": "Purchase Volume",
            "AOV": "Average Order Value",
            "Mean_UnitPrice": "Preferred Unit Price",
            "Count_Orders": "Purchase Frequency",
            "Count_Products": "Product Diversity",
            "Discount_Affinity": "Discount Affinity",
            "Recency": "Recency",
            "Mean_QuantityPerOrder": "Average Quantity per Order"
        }


        #feature_keys = list(important_features.keys())
        feature_keys = [f for f in important_features.keys() if f in cluster_means.columns]
        data_selected = cluster_means[feature_keys]

        # Normalize data
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Replace column names with English labels
        data_normalized.columns = [important_features[col] for col in data_normalized.columns]

        # Setup angles
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        if not cluster_names:
            cluster_names = [f"Cluster {i}" for i in range(k)]

        # Subplot layout
        if k == 4:
            nrows, ncols = 2, 2
            figsize = (12, 10)
        else:
            nrows, ncols = 1, k
            figsize = (5 * k, 5)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, subplot_kw=dict(projection="polar")
        )

        # Ensure axes is always 2D array
        if k == 1:
            axes = np.array([[axes]])
        elif k == 4:
            pass  # already 2D
        else:
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            if k == 4:
                row_idx, col_idx = idx // 2, idx % 2
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[0, idx] if len(axes.shape) == 2 else axes[idx]

            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=3,
                label=cluster_name,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.25, color=color)

            # Styling
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11, weight="bold", color="#2C3E50")
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=9, color="#7F8C8D")
            ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
            ax.set_facecolor("#FAFAFA")

            ax.set_title(
                f"{cluster_name}\n({cluster_means.index[idx]})",
                size=13,
                weight="bold",
                pad=20,
                color=color,
            )

        plt.suptitle(f"Detailed Analysis of Each Cluster (K={k})", size=16, weight="bold", y=1.05)
        plt.tight_layout()
        plt.show()
        
        
        
    def create_radar_chart(self, k, cluster_names=None):
        """
        Create professional radar chart for cluster analysis.

        Args:
            k (int): Number of clusters
            cluster_names (list): Custom names for clusters
        """
        cluster_means = self.cluster_results[k]["means"]

        # Select important features for radar chart
        important_features = {
            "Sum_Quantity": "Purchase Volume",
            "AOV": "Average Order Value",
            "Mean_UnitPrice": "Preferred Unit Price",
            "Count_Orders": "Purchase Frequency",
            "Count_Products": "Product Diversity",
            "Discount_Affinity": "Discount Affinity",
            "Recency": "Recency",
            "Mean_QuantityPerOrder": "Average Quantity per Order"
        }



        # Filter and normalize data
        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Global normalization
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Replace column names with English labels
        data_normalized.columns = [important_features[col] for col in data_normalized.columns]

        # Setup radar chart
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Colors
        colors = (
            ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"] if k == 4 else ["#E74C3C", "#2ECC71", "#3498DB"]
        )
        if not cluster_names:
            cluster_names = [f"Cluster {i}" for i in range(k)]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=4,
                label=cluster_name,
                color=color,
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.15, color=color)

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, weight="bold", color="#2C3E50")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=10, color="#7F8C8D")
        ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
        ax.set_facecolor("#FAFAFA")

        ax.set_title(
            f"Customer Segmentation Analysis (K={k})",
            size=16,
            weight="bold",
            pad=30,
            color="#2C3E50",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

        plt.tight_layout()
        plt.show()



    def train_surrogate_model(self, k):
        """
        Train a RandomForest classifier to mimic the KMeans algorithm.
        This model can be used for SHAP-based explanation analysis.

        Args:
            k (int): Number of clusters

        Returns:
            dict: Training results including model and metrics
        """
        if k not in self.cluster_results:
            raise ValueError(f"Cluster results for k={k} not found. Run apply_kmeans first.")

        # Select feature columns
        feature_cols = [col for col in self.df_scaled.columns if not col.startswith('Cluster_')]
        X = self.df_scaled[feature_cols].values
        y = self.cluster_results[k]['labels']

        # Train RandomForest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)

        # Predictions
        y_pred = rf_model.predict(X)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)

        # Save results
        self.surrogate_models[k] = {
            'model': rf_model,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_cols
        }

        # Print results
        print(f"TRAINING SURROGATE MODEL (k={k})")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"\nThe model can predict clusters {'ACCURATELY' if accuracy >= 0.95 else 'REASONABLY'}.")

        return self.surrogate_models[k]



    def calculate_shap_values(self, k):
        """
        Calculate SHAP values for explaining clustering results using the full dataset.

        Args:
            k (int): Number of clusters

        Returns:
            dict: SHAP explainer and values
        """
        if k not in self.surrogate_models:
            raise ValueError(f"Surrogate model for k={k} not found. Please run train_surrogate_model first.")
        
        # Get model and features
        rf_model = self.surrogate_models[k]['model']
        feature_cols = self.surrogate_models[k]['feature_names']
        X = self.df_scaled[feature_cols].values
        
        # Create SHAP explainer using full dataset as background
        print(f"Calculating SHAP values for {len(X):,} customers...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values_raw = explainer.shap_values(X)
        
        # Convert to list format for multi-class case
        if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            # Multi-class: transpose to (n_classes, n_samples, n_features)
            shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
        else:
            # Binary or single-class
            shap_values = shap_values_raw
        
        # Save results
        self.shap_results[k] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_cols,
            'X': X
        }
        
        print(f"Completed! SHAP values: {len(shap_values)} clusters, each cluster shape: {shap_values[0].shape}")
        return self.shap_results[k]



    def plot_shap_summary(self, k, cluster_id=None):
        """
        Plot SHAP summary (beeswarm) for cluster analysis.

        Args:
            k (int): Number of clusters
            cluster_id (int, optional): Specific cluster to visualize. If None, shows all.
        """
        if k not in self.shap_results:
            raise ValueError(f"SHAP values for k={k} not found. Please run calculate_shap_values first.")
        
        shap_values = self.shap_results[k]['shap_values']
        X = self.shap_results[k]['X']
        feature_names = self.shap_results[k]['feature_names']

        # Plot each cluster or specific cluster
        clusters_to_plot = [cluster_id] if cluster_id is not None else range(k)
        for i in clusters_to_plot:
            shap.summary_plot(
                shap_values[i],
                X,
                feature_names=feature_names,
                max_display=3,
                show=True
            )



    def save_clusters(self, output_dir="../data/processed"):
        """
        Save cluster assignments.

        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        for k in self.cluster_results.keys():
            cluster_col = f"Cluster_{k}"
            cluster_output = self.df_original[[cluster_col]].copy()
            cluster_output.columns = ["Cluster"]
            cluster_output = cluster_output.reset_index()
            cluster_output = cluster_output.sort_values(["Cluster", "customer_id"])

            cluster_output.to_csv(
                f"{output_dir}/customer_clusters_k{k}.csv", index=False
            )
            print(
                f"Cluster assignment for k={k} saved to: {output_dir}/customer_clusters_k{k}.csv"
            )



class DataVisualizer:
    """
    A class for creating visualizations for customer segmentation analysis.

    This class provides methods for plotting various aspects of the data
    including temporal patterns, customer behavior, and cluster analysis.
    """

    def __init__(self):
        """Initialize the DataVisualizer with plotting settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")



    # Plot revenue over time
    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with event_date and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["event_date"].dt.date)["net_revenue_local"].sum()
        daily_revenue.plot()
        plt.title("Daily Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()

        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="event_date", freq="M"))["net_revenue_local"].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        

    # Plot time patterns
    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day of week and hour of day.

        Args:
            df (pd.DataFrame): Dataframe with DayOfWeek and HourOfDay columns
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Purchase Activity by Day and Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week (0=Monday, 6=Sunday)")
        plt.tight_layout()
        plt.show()



    # Plot product analysis
    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe
            top_n (int): Number of top products to show
        """
        # Top products by quantity
        plt.figure(figsize=(12, 5))
        top_products = df.groupby("product_name")["quantity"].sum().sort_values(ascending=False).head(top_n)
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} Products by Quantity Sold")
        plt.xlabel("Quantity Sold")
        plt.tight_layout()
        plt.show()

        # Top products by revenue
        plt.figure(figsize=(12, 5))
        top_revenue_products = df.groupby("product_name")["net_revenue_local"].sum().sort_values(ascending=False).head(top_n)
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} Products by Revenue")
        plt.xlabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()



    # Plot customer distribution
    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.

        Args:
            df (pd.DataFrame): Transaction dataframe
        """
        # Transactions per customer
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("customer_id")["event_id"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Distribution of Transactions per Customer")
        plt.xlabel("Number of Transactions")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()

        # Spending per customer
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("event_id")["net_revenue_local"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Distribution of Total Spending per Customer")
        plt.xlabel("Total Spending (GBP)")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()



    # Plot RFM analysis
    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data (pd.DataFrame): RFM dataframe
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Recency Distribution (Days since last purchase)")
        axes[0].set_xlabel("Days")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Frequency Distribution (Number of Transactions)")
        axes[1].set_xlabel("Number of Transactions")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2])
        axes[2].set_title("Monetary Distribution (Total Spending)")
        axes[2].set_xlabel("Total Spending (GBP)")

        plt.tight_layout()
        plt.show()
