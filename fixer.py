import pandas as pd
import os

# Input CSV path
csv_path = 'data/driving_log.csv'

# Load the CSV
df = pd.read_csv(csv_path)

# Fix each column: 'center', 'left', 'right'
for col in ['center', 'left', 'right']:
    df[col] = df[col].apply(lambda x: os.path.join('IMG', os.path.basename(str(x).strip())))

# Save the cleaned CSV
df.to_csv(csv_path, index=False)

print("✅ driving_log.csv paths have been converted to relative paths like 'IMG/*.jpg'")
