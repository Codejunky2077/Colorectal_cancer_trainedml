#to conduct exploratory data analysis on the dataset
from randomforestml import random_forest_model,x

import pandas as pd
#importing preprocessed dataset 
crc=pd.read_csv("Gut_Microbiome_CRC_Dataset.csv")
print(crc["Disease_status"].value_counts(""))
print(crc.head())
print(crc.describe())
print(crc.shape)
print(crc.info())

#plotting the class distribution of healthy and crc samples in the dataset
import matplotlib.pyplot as plt

crc["Disease_status"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Disease Status")
plt.ylabel("Number of Samples")
plt.show()


import seaborn as sns
random_forest_model()
corr = x.iloc[:, :30].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr)
plt.show()

#PCA plotting
import pandas as pd
crc = pd.read_csv("Gut_Microbiome_CRC_Dataset.csv")
X = crc.drop(columns=["Disease_status","Sample_ID"])
y = crc["Disease_status"]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[y_encoded==0,0],
    X_pca[y_encoded==0,1],
    label="Healthy",
    alpha=0.7
)

plt.scatter(
    X_pca[y_encoded==1,0],
    X_pca[y_encoded==1,1],
    label="CRC",
    alpha=0.7
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Principal Component Analysis (PCA)")

plt.legend()

plt.tight_layout()

plt.show()