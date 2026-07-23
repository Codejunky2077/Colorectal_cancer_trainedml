def xgb():
    import pandas as pd
    #loading dataset
    crc=pd.read_csv("Gut_Microbiome_CRC_Dataset.csv")
    #importing XGBoost 
    from xgboost import XGBClassifier
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

    #creating empty XGBoost model
    xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
    )

    #feeding datset to empty model
    xgb_model.fit(x_train, y_train)

    #predicting results
    y_pred = xgb_model.predict(x_test)

    print("Predictions:", y_pred[:10])
    print("Actual",y_test[:10])

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

    #to check important biomarkers we can use feature importance of the model
    importance = pd.DataFrame({
    "Microbial_Species": x.columns,
    "Importance": xgb_model.feature_importances_
    })

    importance = importance.sort_values(
    by="Importance",
    ascending=False
    )

    print("\nTop 10 Important Microbial Species")
    print(importance.head(10))
    
    #plotting roc-auc curve to check the performance of the model
    from sklearn.metrics import roc_curve, auc
    global y_prob_xgb, fpr_xgb, tpr_xgb, auc_xgb
    y_prob_xgb = xgb_model.predict_proba(x_test)[:,1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
xgb()