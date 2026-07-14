#importing pandas and loading dataset
import pandas as pd
file="CRC_ml/Healthy and crc.xlsx"
#creating categories
healthy=pd.read_excel(file,sheet_name="CTR")
crc=pd.read_excel(file,sheet_name="CRC")
#checking dataset
#print(healthy.head())
#print(crc.head())

#transposing dataset columns and rows to make it suitable for machine learning
healthy = healthy.set_index(healthy.columns[0]).T
crc = crc.set_index(crc.columns[0]).T
#print(healthy.head())
#print(crc.head())

#adding new column to label healthy and crc samples
healthy["Disease_status"]="Healthy"
crc["Disease_status"]="CRC"

#print(healthy.head())
#print(crc.head())

#creating a combined dataset suitable for ml models
dataset=pd.concat([healthy, crc])
print(dataset)

#making sample id as new columns 
dataset.reset_index(inplace=True)
dataset.rename(columns={"index": "Sample_ID"}, inplace=True)
#print(dataset)

#filling null values with 0 to avoid errors in machine learning models and checking again
dataset = dataset.fillna(0)
print("Remaining missing values:",
      dataset.isna().sum().sum())
print(dataset.head())
#obtaing preprocessed dataset in csv format ready to be used for machine learning models
dataset.to_csv(
    r"CRC_ml/Gut_Microbiome_CRC_Dataset.csv",
    index=False
)
print("Dataset successfully preprocessed and saved.")