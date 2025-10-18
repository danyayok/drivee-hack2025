import pandas as pd
df = pd.read_csv('train.csv')
print(df.head(10).to_string())
print("\nКолонки:", df.columns.tolist())
print("Размер:", df.shape)