# src/preprocess.py
import pandas as pd
import os

def preprocess_data(input_path: str, output_path: str):
    print(f"Lade Rohdaten von: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Duplikate entfernen
    df.drop_duplicates(inplace=True)
    
    # 2. Feature Engineering
    df['Is_Holiday_Season'] = df['Month'].isin(['Nov', 'Dec']).astype(int)
    df['Total_Clicks'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    
    # 3. Formatierung (Boolean zu Int)
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # 4. One-Hot-Encoding
    df_clean = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)
    
    # 5. Speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Reproduzierbarer Datensatz gespeichert unter: {output_path}")
    print(f"Form des finalen Datensatzes: {df_clean.shape}")

if __name__ == "__main__":
    # Pfade relativ zum Skript auflösen
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IN_PATH = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")
    OUT_PATH = os.path.join(BASE_DIR, "data", "processed_online_shoppers.csv")
    
    preprocess_data(IN_PATH, OUT_PATH)