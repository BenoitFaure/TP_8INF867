# TP1 - Algorithmes de réduction de dimentionalité

# Imports
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# -----------------DATAPROCESSING-----------------
print("\n 1] DATAPROCESSING \n")

# a) Load data
df = pd.read_csv('train_diab.csv')
target = 'Diabetes_binary'
print(df.head(2))

# b) Missing values
df.fillna(0, inplace=True) # Replace missing values by 0

# c) Categorical variable
#   No categorical variables here

# d) Scaling
cols_to_scale = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
df[cols_to_scale] = MinMaxScaler().fit_transform(df[cols_to_scale])

# e) Extract features and target values
X, y = df.drop(columns=[target]), df[[target]]

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

    # vp_tr = sorted_eigenvectors.transpose()
    return _X @ sorted_eigenvectors

pca_exec_time = time.time()

X_pca = my_pca(X, num_feat=num_features).values

pca_exec_time = time.time() - pca_exec_time

print(f"Done in {pca_exec_time} s")

# c) SelectKBest ---------------------------------
print("\n c) SelectKBest (chi2)")

skb_exec_time = time.time()

skb_obj = SelectKBest(chi2, k=num_features)
X_skb = skb_obj.fit_transform(X, y)

skb_exec_time = time.time() - skb_exec_time

print(f"Done in {skb_exec_time} s")
print(skb_obj.get_feature_names_out(X.columns))

# -------------------COMPARISON-------------------
print("\n 3] COMPARISON \n")

num_elements = 500
y_red = y[target][:num_elements]
print(f"Running TSNE on {num_elements} elements")

# Do clustering for 2d visulaisation
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne_pca_sk = tsne.fit_transform(X_pca_sk[:num_elements, :])
print("tsne pca sk done")
X_tsne_pca = tsne.fit_transform(X_pca[:num_elements, :])
print("tsne pca done")
X_tsne_skb = tsne.fit_transform(X_skb[:num_elements, :])
print("tsne skb done")

# Set up fig
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
ax1, ax2, ax3 = axs

def plot_map(ax, dim1, dim2, y_data, title):
    scatter = ax.scatter(dim1, dim2, c=y_data, cmap='viridis', marker='o', s=50)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Regression Target ({target})')

# Visualise tsne for PCA sklearn
plot_map(ax1, X_tsne_pca_sk[:, 0], X_tsne_pca_sk[:, 1], y_red, 't-SNE Visualization for PCA sklearn')

# Visualise tsne for myPCA
plot_map(ax2, X_tsne_pca[:, 0], X_tsne_pca[:, 1], y_red, 't-SNE Visualization for PCA')

# Visualise tsne for SelectKBest
plot_map(ax3, X_tsne_skb[:, 0], X_tsne_skb[:, 1], y_red, 't-SNE Visualization for SelectKBest (chi2)')

# Show
plt.tight_layout()
plt.show()