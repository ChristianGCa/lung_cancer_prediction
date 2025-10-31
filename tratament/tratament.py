import pandas as pd

def preprocess_lung_cancer_data(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    columns_to_map = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'COUGHING',
        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]

    for col in columns_to_map:
        if col in df.columns:
            df[col] = df[col].map({1: 0, 2: 1})

    if 'LUNG_CANCER' in df.columns:
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

    df.to_csv(output_csv_path, index=False)
    print(f"Dados salvos em {output_csv_path}")

input_file = '/home/christian/Documentos/lung_cancer.csv'
output_file = '/home/christian/Documentos/lung_cancer_processed.csv'

preprocess_lung_cancer_data(input_file, output_file)
