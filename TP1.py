# TP1 - Algorithmes de réduction de dimentionalité

# Imports
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# -----------------DATAPROCESSING-----------------
print("\n 1] DATAPROCESSING \n")

# a) Load data
df = pd.read_csv('train.csv')
print(df.head(2))


# b) Missing values
df.fillna(0, inplace=True) # Replace missing values by 0

# c) Categorical variable
df_dummies = pd.get_dummies(df) # One hot encoding

# d) Extract features and target values
X, y = df_dummies.drop(columns=['Sales']), df_dummies[['Sales']]
y_log = np.log(y['Sales']) #Scale the Sales to make it more uniform

# ------------DIMENSIONALITY REDUCTION------------
print("\n 2] DIMENSIONALITY REDUCTION \n")

num_features = 10
print(f"Reducing to {num_features} features")

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
    sorted_eigenvalue = eigen_values[sorted_index]

    #similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    if num_feat:
        sorted_eigenvectors = sorted_eigenvectors[:,:num_feat]

    vp_tr = sorted_eigenvectors.transpose()
    return vp_tr * _X

pca_exec_time = time.time()

X_pca = X_pca_sk # my_pca(X, num_feat=num_features)

pca_exec_time = time.time() - pca_exec_time

print(f"Done in {pca_exec_time} s")

# c) SelectKBest ---------------------------------
print("\n c) SelectKBest (f_regression)")

skb_exec_time = time.time()

skb_obj = SelectKBest(f_regression, k=num_features)
X_skb = skb_obj.fit_transform(X, y_log)

skb_exec_time = time.time() - skb_exec_time

print(f"Done in {skb_exec_time} s")
print(skb_obj.get_feature_names_out(X.columns))

# -------------------COMPARISON-------------------
print("\n 3] COMPARISON \n")

# Do clustering for 2d visulaisation
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne_pca_sk = tsne.fit_transform(X_pca_sk)
X_tsne_pca = tsne.fit_transform(X_pca)
X_tsne_skb = tsne.fit_transform(X_skb)

# Set up fig
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
ax1, ax2, ax3 = axs

def plot_map(ax, dim1, dim2, y_data, title):
    scatter = ax.scatter(dim1, dim2, c=y_data, cmap='viridis', marker='o', s=50)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Regression Target (log(Sales))')

# Visualise tsne for PCA sklearn
plot_map(ax1, X_tsne_pca_sk[:, 0], X_tsne_pca_sk[:, 1], y_log, 't-SNE Visualization for PCA sklearn')

# Visualise tsne for myPCA
plot_map(ax2, X_tsne_pca[:, 0], X_tsne_pca[:, 1], y_log, 't-SNE Visualization for PCA')

# Visualise tsne for SelectKBest
plot_map(ax3, X_tsne_skb[:, 0], X_tsne_skb[:, 1], y_log, 't-SNE Visualization for SelectKBest')

# Show
plt.tight_layout()
plt.show()