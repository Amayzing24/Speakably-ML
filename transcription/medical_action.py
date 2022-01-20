import pandas as pd

df = pd.read_csv("data.csv", encoding="ISO-8859-1")

df = df[df['Tag'] == 6]

df = df.reset_index()

df.to_csv("medical_action.csv")
