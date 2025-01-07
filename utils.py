import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn

def fill_missing_values(df, object_columns, float_columns):
   df[object_columns] = df[object_columns].fillna(method = 'pad')
   df[float_columns] = df[float_columns].apply(lambda col: col.fillna(col.mean()))
   return df

def encode_scale_df(df, object_columns, float_columns):
    for column in object_columns:
       df[column] = LabelEncoder().fit_transform(df[column])
    df[float_columns] = StandardScaler().fit_transform(df[float_columns])
    return df

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # İlk katman
        self.fc2 = nn.Linear(128, 64)          # İkinci katman
        self.fc3 = nn.Linear(64, num_classes)  # Çıkış katmanı

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)
    
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
       y_val_pred = model(X_val)
       y_val_pred = torch.argmax(y_val_pred, dim=1)
       accuracy = (y_val_pred == y_val).float().mean()
       print(f'Validation Accuracy: {accuracy.item():.4f}')