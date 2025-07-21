import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

file_path = '/Users/admin/Desktop/RandomForest/BL-1Notes_2Round_Sep2024.xlsx'

# Step 1: Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Step 2: Inspect the first few rows to understand the data structure
print("First few rows of the dataset:")
print(df.head())

# Step 3: Drop the ID and non-numeric columns
# Store the ID columns to keep them for reference
df_id = df[['Total Number of samples', 'Identification 1st Round', 'Identification Erica']]  # Store the ID columns
df_clean = df.drop(columns=['Total Number of samples', 'Identification 1st Round', 'Identification Erica'])

# Step 4: Convert all columns to numeric, non-numeric values will be coerced to NaN
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# Step 5: Drop columns that have all NaN values (like 'Unnamed: 3' or any other empty column)
df_clean = df_clean.dropna(axis=1, how='all')

# Debugging: Check if any columns were dropped
print(f"Columns after dropping columns with all NaN values: {df_clean.columns}")

# Step 6: Select only numeric columns for PCA
df_clean = df_clean.select_dtypes(include=[np.number])

# Debugging: Check if there are any numeric columns selected
print(f"Columns selected for PCA: {df_clean.columns}")

# Check if df_clean is empty
if df_clean.empty:
    raise ValueError("No numeric columns found in the dataset after cleaning.")

# Step 7: Handle missing values by imputing with the mean
# Using SimpleImputer to fill in missing values with the column mean
imputer = SimpleImputer(strategy='mean')  # Impute with the column mean
df_clean_imputed = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

# Debugging: Check if there are any NaN values after imputation
print(f"Missing values after imputation: {df_clean_imputed.isnull().sum()}")

# Step 8: Standardize the Data (important for PCA)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean_imputed)

# Step 9: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(df_scaled)

# Step 10: Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 11: Plot the PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7, s=100)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 12: Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio of each component: {explained_variance}")
print(f"Total Variance Explained by first 2 components: {explained_variance.sum()}")

# Step 13: Plot the explained variance ratio
plt.figure(figsize=(8, 6))
sns.barplot(x=['PC1', 'PC2'], y=explained_variance)
plt.title('Explained Variance Ratio of Principal Components')
plt.ylabel('Variance Explained')
plt.show()

# Step 14: Optional - Visualize the correlation between features and principal components without numbers
plt.figure(figsize=(8, 6))
sns.heatmap(pca.components_, cmap='coolwarm', annot=False, xticklabels=df_clean_imputed.columns, yticklabels=['PC1', 'PC2'])
plt.title('Feature Correlation with Principal Components')
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.show()

# Calculate correlation matrix
correlation_matrix = df_clean_imputed.corr()

# Plot a heatmap of the correlation matrix without numbers
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Visualize clusters in PCA space
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
plt.title('K-Means Clusters in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Fit Random Forest model and get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df_clean_imputed, target)

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': df_clean_imputed.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Random Forest')
plt.show()
