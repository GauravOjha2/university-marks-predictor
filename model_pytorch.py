import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load and clean data
# ... existing code ...
df = pd.read_csv('Book1.csv')
df.columns = df.columns.str.strip()

X_train = torch.tensor(df[['scale_last']].values, dtype=torch.float32)
y_train = torch.tensor(df['marks_lst'].values, dtype=torch.float32).view(-1, 1)

print("Training targets (marks_lst):", y_train.flatten().numpy())

# Define a simple neural network with 1 hidden layer (like Keras Sequential)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()

# Use Adam optimizer and MSE loss (like Keras)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    # Optionally print loss every 500 epochs
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Predict for new data
X_new = torch.tensor(df[['scale_now']].values, dtype=torch.float32)
model.eval()
predicted_marks = model(X_new).detach().numpy().flatten()

for i, (subject, scale_now, pred) in enumerate(zip(df['subjects'], df['scale_now'], predicted_marks)):
    print(f"Subject: {subject}, This semester's self-assessment: {scale_now}, Predicted marks: {pred:.2f}")

df['predicted_marks_this_sem'] = predicted_marks
