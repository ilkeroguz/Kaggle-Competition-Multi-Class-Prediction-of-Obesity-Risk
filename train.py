import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import optim
from sklearn.model_selection import train_test_split
from utils import fill_missing_values, encode_scale_df, MultiLayerPerceptron, evaluate_model

train_set = pd.read_csv('data/train.csv')

object_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
float_columns = ['id', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
print(train_set['NObeyesdad'].unique())

train_set = fill_missing_values(train_set, object_columns, float_columns)
#print(train_set.isnull().sum())
train_set = encode_scale_df(train_set, object_columns, float_columns)

train_tensor = torch.tensor(train_set.values, dtype= torch.float32)

input_size = train_tensor.shape[1] - 1
num_classes = 7
model = MultiLayerPerceptron(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
X = train_tensor[:, : -1]
y = train_tensor[:, -1].long()

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size= 0.2,
                                                  stratify= y,
                                                  random_state= 42
)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(X_train)
    
    loss = criterion(y_pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
evaluate_model(model, X_val, y_val)

torch.save(model.state_dict(), 'Obesity_Preditcion.pth')