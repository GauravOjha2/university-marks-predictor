import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('Book1.csv')
df.columns = df.columns.str.strip() 

X_train = df[['scale_last']].values
y_train = df['marks_lst'].values

print("Training targets (marks_lst):", y_train)  

model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=2000, verbose=0)  

X_new = df[['scale_now']].values
predicted_marks = model.predict(X_new)

for i, (subject, scale_now, pred) in enumerate(zip(df['subjects'], df['scale_now'], predicted_marks.flatten())):
    print(f"Subject: {subject}, This semester's self-assessment: {scale_now}, Predicted marks: {pred:.2f}")

df['predicted_marks_this_sem'] = predicted_marks