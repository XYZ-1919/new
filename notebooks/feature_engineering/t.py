import pandas as pd 

df = pd.read_csv(r"C:\Users\91786\Desktop\DS\food_ds\data\extracted\train.csv")

print(df.isnull().sum())