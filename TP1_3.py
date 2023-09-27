# TP1 - Algorithmes de réduction de dimentionalité

# Imports
import time
import numbers

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# -----------------DATAPROCESSING-----------------
print("\n 1] DATAPROCESSING \n")

# a) Load data
df = pd.read_csv('train.csv')
print(df.head(2))

target = 'Sales'
y = df[[target]]
df = df.drop(target, axis=1)

# b) Missing values
df.fillna(0, inplace=True) # Replace missing values by 0

# c) Uneced columns
df = df.drop(["Row ID"], axis=1)

# d) Extract numeric and categorical values
categ_var = ["Postal Code"]
for column in df.columns:
    if df[column].dtype == 'object':  
       categ_var.append(column)

# c) Categorical variable
if not len(categ_var) == 0:
    encoder = OneHotEncoder(sparse_output=False,min_frequency=10)
    categ_encoded = encoder.fit_transform(df[categ_var])
    df = df.drop(categ_var, axis=1)

# d) Scaling
if not df.empty:
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

# f) Concat all
if not len(categ_var) == 0:
    df = pd.concat([df, pd.DataFrame(categ_encoded, columns=encoder.get_feature_names_out(categ_var))], axis=1)

# e) Extract features and target values
X, y = df, np.log(y)

# ------------DIMENSIONALITY REDUCTION------------
print("\n 2] DIMENSIONALITY REDUCTION \n")

num_features = 100
print(f"Reducing to {num_features} features from {df.columns.size}")

# a) PCA - sklearn  ------------------------------
print("\n a) PCA - sklearn")

pca_sk_exec_time = time.time()

pca_sk_obj = PCA(n_components=num_features)
X_pca_sk = pca_sk_obj.fit_transform(X)

pca_sk_exec_time = time.time() - pca_sk_exec_time

print(f"Done in {pca_sk_exec_time} s")

# b) PCA -----------------------------------------
print("\n b) PCA")

def my_pca(_X, num_feat=None): # TODO: This is too expensive
    X_meaned = _X - np.mean(_X , axis = 0)
    z = X_meaned.cov()
    eigen_values , eigen_vectors = np.linalg.eig(z)

    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]

    #similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    if num_feat:
        sorted_eigenvectors = sorted_eigenvectors[:,:num_feat]

    return _X @ sorted_eigenvectors.real

pca_exec_time = time.time()

X_pca = my_pca(X, num_feat=num_features).values

pca_exec_time = time.time() - pca_exec_time

print(f"Done in {pca_exec_time} s")

# c) SelectKBest ---------------------------------
print("\n c) SelectKBest (chi2)")

skb_exec_time = time.time()

skb_obj = SelectKBest(f_regression, k=num_features)
X_skb = skb_obj.fit_transform(X, y)

skb_exec_time = time.time() - skb_exec_time

print(f"Done in {skb_exec_time} s")
print(skb_obj.get_feature_names_out(X.columns))

# -------------------COMPARISON-------------------
print("\n 3] COMPARISON \n")

num_elements = y.values.shape[0]
y_red = y[target][:num_elements]
print(f"Running TSNE on {num_elements} elements")

# Do clustering for 2d visulaisation
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne_base = tsne.fit_transform(X.values[:num_elements, :])
print("tsne base done")
X_tsne_pca_sk = tsne.fit_transform(X_pca_sk[:num_elements, :])
print("tsne pca sk done")
X_tsne_pca = tsne.fit_transform(X_pca[:num_elements, :])
print("tsne pca done")
X_tsne_skb = tsne.fit_transform(X_skb[:num_elements, :])
print("tsne skb done")

# Set up fig
fig, axs = plt.subplots(2, 2, figsize=(18, 18))
ax1, ax2, ax3, ax4 = axs.flatten()

def plot_map(ax, dim1, dim2, y_data, title):
    scatter = ax.scatter(dim1, dim2, c=y_data, cmap='viridis', marker='o', s=5)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Regression Target ({target})')

# Visualise tsne for no dimensionality reduction
plot_map(ax1, X_tsne_base[:, 0], X_tsne_base[:, 1], y_red, 't-SNE Visualization for no dimensionality reduction')

# Visualise tsne for PCA sklearn
plot_map(ax2, X_tsne_pca_sk[:, 0], X_tsne_pca_sk[:, 1], y_red, 't-SNE Visualization for PCA sklearn')

# Visualise tsne for myPCA
plot_map(ax3, X_tsne_pca[:, 0], X_tsne_pca[:, 1], y_red, 't-SNE Visualization for PCA')

# Visualise tsne for SelectKBest
plot_map(ax4, X_tsne_skb[:, 0], X_tsne_skb[:, 1], y_red, 't-SNE Visualization for SelectKBest (f_regression)')

# Show
plt.tight_layout()
plt.show()