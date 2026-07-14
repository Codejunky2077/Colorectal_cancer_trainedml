#using pandas to get info about datatype of file 
import pandas as pd
sample_file=("Gut_Microbiome_CRC_Dataset.csv")
crc=pd.read_csv(sample_file)
print(crc.info())

#we start to labelencode the data of healthy and crc information for decision tree model
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
crc["Disease_status"]=le.fit_transform(crc["Disease_status"])

#splitting the dataset into two 
x=crc.drop(columns=["Sample_ID","Disease_status"])
y=crc["Disease_status"]

#using train_test_split to split the dataset into train and 20% of data to be used as test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=40,stratify=y
)

#we are now checking the split of data in train and test data
print(f"Train data shape in X {x_train.shape}")
print(f"Test data shape in X {x_test.shape}")
print(f"Train data shape in y {y_train.shape}")
print(f"Test data shape in y {y_test.shape}")

#we create empty decision tree model and fit the train data to it
from sklearn.tree import DecisionTreeClassifier
crc_model=DecisionTreeClassifier()

#we fit the data 
crc_model.fit(x_train,y_train)

#now we will predict the test data using the model
y_pred=crc_model.predict(x_test)
print(y_pred[:10])
print(y_test[:10])

#now we analyse the prediction accuracy and confusion matrix to check the performance of the model
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy score of model is {accuracy*100 :.2f}%")

#confusion matrix to check the performance of the model
confuse=confusion_matrix(y_test,y_pred)
print("Confusion matrix is:")
print(confuse)

#classification report to check the performance of the model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))