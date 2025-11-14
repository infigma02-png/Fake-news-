import pandas as pd

# def load_dataset(file_path):
#     """Loads the dataset and returns a DataFrame."""
#     df = pd.read_csv(file_path)
#     print(f"✅ Dataset loaded successfully from: {file_path}")
#     print(f"Total Records: {df.shape[0]}, Columns: {list(df.columns)}")
#     print(df.head(), "\n")
#     return df
  
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
    print(f"Total records: {len(df)}")
    return df