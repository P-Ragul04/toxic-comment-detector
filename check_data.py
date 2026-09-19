import pandas as pd

train_df = pd.read_csv('./dataset/train.csv')
print(train_df.shape)
print(train_df.columns.tolist())
print(train_df.iloc[0])