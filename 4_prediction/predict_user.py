import torch
import torch.nn as nn

class LungCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Carregar modelo
model = LungCancerModel()
model.load_state_dict(torch.load("lung_cancer_model.pth"))
model.eval()

def ask(text):
    return float(input(f"{text} (0 = Não, 1 = Sim): "))

gender = input("Gênero (Masculino/Feminino): ")
gender = 0 if gender.lower().startswith("m") else 1

age = float(input("Idade: "))
smoking = ask("Fuma?")
yellow_fingers = ask("Dedos amarelados?")
anxiety = ask("Ansiedade?")
chronic_disease = ask("Doença crônica?")
fatigue = ask("Fadiga?")
allergy = ask("Alergia?")
wheezing = ask("Chiado no peito?")
peer_pressure = ask("Pressão de colegas? (Peer Pressure)")
alcohol = ask("Bebe álcool?")
coughing = ask("Tosse?")
sob = ask("Falta de ar?")
swallow = ask("Dificuldade pra engolir?")
chest_pain = ask("Dor no peito?")

# Converter para tensor
data = torch.tensor([[gender, age, smoking, yellow_fingers, anxiety, chronic_disease, 
                      fatigue, allergy, wheezing, peer_pressure, alcohol, coughing, sob,
                      swallow, chest_pain]], dtype=torch.float32)

# Prever
with torch.no_grad():
    prob = model(data).item()

print(f"\nProbabilidade estimada: {prob:.2f}")

if prob > 0.5:
    print("Alta chance de desenvolver câncer de pulmão")
else:
    print("Baixa chance de desenvolver câncer de pulmão")
