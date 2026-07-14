import pandas as pd 
from sklearn.svm import SVC
#loading preprocessed dataset
crc=pd.read_csv("Gut_Microbiome_CRC_Dataset.csv")

#we need to encode the isease_status column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
crc["Disease_status"]=le.fit_transform(crc["Disease_status"])

#splitting dataset into features and target variable
x=crc.drop(columns=["Sample_ID","Disease_status"])
y=crc["Disease_status"]

#to train and test the model we need to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20,stratify=y)
#creating empty SVM model
svm_model = SVC(
    kernel="rbf",
    random_state=42
)
#training the ml model
svm_model.fit(x_train,y_train)



#to check predictions
y_pred=svm_model.predict(x_test)
print("Predictions:",y_pred[:10])
print("Actual values:",y_test[:10])


#doing analytics
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

