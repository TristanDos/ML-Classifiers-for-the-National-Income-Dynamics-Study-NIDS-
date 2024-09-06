import pandas as pd
from tqdm import tqdm
from typing import List
import plotter
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

waves_in = input("Which waves should be labelled (separate with spaces eg. '1 3 5')\n")
waves_to_label = waves_in.split(" ")
waves_to_label = [int(wave_str) for wave_str in waves_to_label]

class Wave:
    def __init__(self, data: pd.DataFrame, select_cols: List[str]):
        self.data: pd.DataFrame = data
        self.select_cols: List[str] = select_cols
    
    def __str__(self):
        return str(self.data)

waves: List[Wave] = []

# Selecting column names for CESD-10 Scale related features
cesd_col_names = ["_a_emobth", "_a_emomnd", "_a_emodep", "_a_emoeff", "_a_emohope",
                "_a_emofear", "_a_emoslp", "_a_emohap", "_a_emolone", "_a_emogo"]

# Scoring dictionaries
normal_scoring = {
    'Rarely or none of the time (less than 1 day)': 0,
    'Some or little of the time (1-2 days)': 1,
    'Occasionally or a moderate amount of time (3-4 days)': 2,
    'All of the time (5-7 days)': 3
}

reverse_scoring = {
    'Rarely or none of the time (less than 1 day)': 3,
    'Some or little of the time (1-2 days)': 2,
    'Occasionally or a moderate amount of time (3-4 days)': 1,
    'All of the time (5-7 days)': 0
}

incidence = []

# Loop through each wave
for i in tqdm(waves_to_label, desc="Labelling Participants"):
    url = 'CSV/wave' + str(i) + '_select.csv'
    data = pd.read_csv(url)

    # Header text for each column based on wave
    header = 'w' + str(i)
    
    # Create the select columns
    select_cols = [header + col for col in cesd_col_names]

    # Filter rows based on valid CESD answers
    cesd_valid_answers = ['Rarely or none of the time (less than 1 day)',
                          'Some or little of the time (1-2 days)',
                          'Occasionally or a moderate amount of time (3-4 days)',
                          'All of the time (5-7 days)']
    
    # Only keep rows where all select_cols have valid answers
    # new_data = data[data[select_cols].isin(cesd_valid_answers).all(axis=1)].fillna(data.mode(), inplace=True)
    new_data = data[data[select_cols].isin(cesd_valid_answers).all(axis=1)]
    
    # Apply scoring to all CESD columns
    for idx, col in enumerate(select_cols):
        if idx == 4 or idx == 7:  # Reverse scoring for columns 5 and 8 (0-indexed as 4 and 7)
            new_data[col] = new_data[col].replace(reverse_scoring)
        else:
            new_data[col] = new_data[col].replace(normal_scoring)

    # Derive "Depressed" column: 1 if score >= 10, else 0
    new_data['score'] = new_data[select_cols].sum(axis=1)
    new_data['depressed'] = (new_data['score'] >= 10).astype(int)

    # Check which rows have NaN values
    nan_rows = new_data[new_data.isna().any(axis=1)]

    # Print rows with NaN values
    # print(nan_rows)

    new_data.to_csv(f'CSV/wave{i}_select_labelled.csv', index=False)

    percentage_depressed = plotter.get_percent_na(new_data['depressed'].replace(1, pd.NA))

    incidence.append(percentage_depressed)

    # Append the wave object to the list
    wave = Wave(new_data, select_cols)
    waves.append(wave)

for i in range(len(incidence)): print((f"Wave {i+1}: {round(incidence[i], 3)}% depressed"))