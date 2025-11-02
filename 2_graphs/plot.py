import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/christian/Documentos/lung_cancer_processed.csv")
df["GENDER"] = df["GENDER"].map({"F": "Feminino", "M": "Masculino"})

print(df)

counts = df.groupby(["GENDER", "LUNG_CANCER"]).size().unstack(fill_value=0)
total_by_gender = counts.sum(axis=1)
proportions = counts.div(total_by_gender, axis=0)
cancer_proportion = proportions[1]

cancer_proportion.plot(kind="bar", figsize=(8, 5))
plt.title("Proporção de Câncer de Pulmão por Gênero")
plt.xlabel("Gênero")
plt.ylabel("Proporção de Câncer de Pulmão")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

df["SMOKING"] = df["SMOKING"].map({1: "Fuma", 0: "Não Fuma"})

smoking_gender_counts = df.groupby(["GENDER", "SMOKING"]).size().unstack(fill_value=0)
smoking_gender_proportions = smoking_gender_counts.div(smoking_gender_counts.sum(axis=1), axis=0)

smoking_gender_proportions.plot(kind="bar", figsize=(10, 6))
plt.title("Proporção de Fumantes e Não Fumantes por Gênero")
plt.xlabel("Gênero")
plt.ylabel("Proporção")
plt.xticks(rotation=0)
plt.legend(title="Tabagismo")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

