import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

names = ['Type', 'Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Pov', 'Vanilla', 'Salt']

df = pd.read_csv('recipes_muffins_cupcakes_scones.csv')
print(df)

X = np.array(df.iloc[:, 1:9])
y = np.array(df['Type'])

print(f'Before Standrdization: {X}')
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print(f'Variance: {explained_variance}')

cumulative_variance = np.cumsum(explained_variance)
print(f'Cumulative variance: {cumulative_variance}')

plt.plot(range(1,8), cumulative_variance, color='blue')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('PC = 1-8')
plt.show()

for i in range(len(y)):
    if y[i] == 'Muffin':
        y[i] = 1
    elif y[i] == 'Cupcake':
        y[i] = 2
    else:
        y[i] = 3
print(y)

#scatter Plot
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap='plasma')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot between PC1 and PC2')
plt.show()

#Histogram Plot
fig,axes =plt.subplots(2,4, figsize=(15, 12))
ax=axes.ravel()
for i in range(8):
    _,bins=np.histogram(X[:,i],bins=25)
    ax[i].hist(X[y == 1,i],bins=bins,color='r',alpha=0.5, label='Muffin')
    ax[i].hist(X[y == 2,i],bins=bins,color='g',alpha=0.5, label='Cupcake')
    ax[i].hist(X[y == 3,i],bins=bins,color='b',alpha=0.5, label='Scone')
    ax[i].set_title(names[i + 1],fontsize=9)
    ax[i].axes.get_xaxis().set_visible(False) 
    ax[i].set_yticks(())
    ax[0].legend(loc='best',fontsize=8)
plt.tight_layout()
plt.show()

# Heatmap of features with largest variation in PC1 and PC2
pc_df = pd.DataFrame(principalComponents[:, :2], columns=['PC1', 'PC2'])
sns.heatmap(pc_df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Features with Largest Variation in PC1 and PC2')
plt.show()

# Find features with highest and lowest variation in PC1 and PC2
pc1_components = pca.components_[0]
pc2_components = pca.components_[1]

max_pc1_feature = df.columns[1:][np.argmax(pc1_components)]
min_pc1_feature = df.columns[1:][np.argmin(pc1_components)]
max_pc2_feature = df.columns[1:][np.argmax(pc2_components)]
min_pc2_feature = df.columns[1:][np.argmin(pc2_components)]

print(f'Feature with highest variation in PC1: {max_pc1_feature}')
print(f'Feature with lowest variation in PC1: {min_pc1_feature}')
print(f'Feature with highest variation in PC2: {max_pc2_feature}')
print(f'Feature with lowest variation in PC2: {min_pc2_feature}')

# Correlation heatmap of original features
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, 1:].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Original Features')
plt.show()
