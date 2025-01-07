import pandas as pd
from utils import fill_missing_values, encode_scale_df, MultiLayerPerceptron
import torch

test_set = pd.read_csv('data/test.csv')

float_columns = ['id','Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
object_columns = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

id_column = test_set['id']
u_test_set = fill_missing_values(test_set, object_columns, float_columns)
print(test_set.isnull().sum())

u_test_set = encode_scale_df(u_test_set, object_columns, float_columns)
print(test_set.head())

test_tensor = torch.tensor(u_test_set.values, dtype=torch.float32)

X = test_tensor

input_size = test_tensor.shape[1]
num_classes = 7
model = MultiLayerPerceptron(input_size, num_classes)

model.load_state_dict(torch.load('Obesity_Preditcion.pth'))
model.eval()

with torch.no_grad():
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, dim=1)
    
obesity_classes = ['Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Obesity_Type_I']

y_pred = y_pred.numpy()  # NumPy array'e dönüştür

print("y_test_pred type:", type(y_pred))  # Veri tipini kontrol edin
print("y_test_pred shape:", y_pred.shape)  # Şekil kontrolü (tensor ise)
print("y_test_pred values:", y_pred)  # İlk birkaç değeri yazdır
print("obesity_classes length:", len(obesity_classes))  # Sınıf listesi uzunluğu

# Sonuçları bir DataFrame'e dönüştürme
result_df = pd.DataFrame({
    'id': id_column,  # id'leri alıyoruz
    'Obesity': [obesity_classes[label] for label in y_pred]  # Tahmin edilen sınıf etiketlerini anlamlı hale getiriyoruz
})

result_df.to_csv('test_predictions.csv', index=False)

print("Test sonuçları kaydedildi: test_predictions.csv")