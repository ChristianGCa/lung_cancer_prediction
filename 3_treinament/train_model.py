import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/christian/Documentos/lung_cancer_processed.csv")

# Converter gênero para número
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1}).astype(float)

df = df.sample(frac=1).reset_index(drop=True)

X = df.drop(columns=["LUNG_CANCER"]).values
y = df["LUNG_CANCER"].values.reshape(-1, 1)

# Fazer transformação para tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split manual 80/20 para treino e teste (treino = 80% e teste = 20%)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

# MODELO DE REDE NEURAL
class LungCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.shape[1], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = LungCancerModel()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

training_losses = []

# Nova lista para armazenar a precisão de treinamento
training_precision = []

# TREINAMENTO
epochs = 200
for epoch in range(epochs):
    for Xb, yb in train_loader:
        y_pred = model(Xb)
        loss = criterion(y_pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    training_losses.append(loss.item())

    # Calcular precisão após cada época
    with torch.no_grad():
        preds = model(X_test)
        preds_class = (preds > 0.5).float()
        
        TP = ((preds_class == 1) & (y_test == 1)).sum().item()
        FP = ((preds_class == 1) & (y_test == 0)).sum().item()
        
        precision_epoch = TP / (TP + FP + 1e-8)
        training_precision.append(precision_epoch)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Precision: {precision_epoch:.4f}")

print("\nTreinamento finalizado")

# ======= AVALIAÇÃO =======
with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).float()

# Matriz de Confusão
TP = ((preds_class == 1) & (y_test == 1)).sum().item()
TN = ((preds_class == 0) & (y_test == 0)).sum().item()
FP = ((preds_class == 1) & (y_test == 0)).sum().item()
FN = ((preds_class == 0) & (y_test == 1)).sum().item()

# Métricas
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)   # Sensibilidade
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

print("\nMATRIZ DE CONFUSÃO")
print("----------------------")
print(f"TP (Verdadeiro Positivo):   {TP}")
print(f"TN (Verdadeiro Negativo):   {TN}")
print(f"FP (Falso Positivo):        {FP}")
print(f"FN (Falso Negativo):        {FN}")

print("\nMÉTRICAS")
print("----------------------")
print(f"Acurácia:   {accuracy:.4f}")
print(f"Precisão:   {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:   {f1:.4f}")

# Gráfico da Perda de Treinamento
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Perda de Treinamento')
plt.title('Perda de Treinamento ao Longo das Épocas')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

# Gráfico da Precisão de Treinamento
plt.figure(figsize=(10, 5))
plt.plot(training_precision, label='Precisão de Treinamento', color='green')
plt.title('Precisão de Treinamento ao Longo das Épocas')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.legend()
plt.grid(True)
plt.savefig('training_precision.png')
plt.show()

# Matriz de Confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.numpy(), preds_class.numpy())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Câncer', 'Câncer'], yticklabels=['Não Câncer', 'Câncer'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.savefig('confusion_matrix.png')
plt.show()

torch.save(model.state_dict(), "lung_cancer_model.pth")
print("\n✅ Modelo salvo em: lung_cancer_model.pth")
