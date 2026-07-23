#to cumulatively generate the AUC-ROC curve for all the models used in this project
from decisiontreeml import fpr_dt, tpr_dt, auc_dt
from randomforestml import fpr_rf, tpr_rf, auc_rf   
from SVMml import fpr_svm, tpr_svm, auc_svm
from XGBoostml import fpr_xgb, tpr_xgb, auc_xgb
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

plt.plot(fpr_dt, tpr_dt,
         label=f"Decision Tree (AUC = {auc_dt:.3f})")

plt.plot(fpr_rf, tpr_rf,
         label=f"Random Forest (AUC = {auc_rf:.3f})")

plt.plot(fpr_svm, tpr_svm,
         label=f"SVM (AUC = {auc_svm:.3f})")

plt.plot(fpr_xgb, tpr_xgb,
         label=f"XGBoost (AUC = {auc_xgb:.3f})")

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Machine Learning Models")

plt.legend(loc="lower right")

plt.grid()

plt.show()

#AUC values of all the models
print("Decision Tree AUC :", auc_dt)
print("Random Forest AUC :", auc_rf)
print("SVM AUC :", auc_svm)
print("XGBoost AUC :", auc_xgb)