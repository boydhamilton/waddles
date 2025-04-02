import pandas as pd

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.rename(columns={
        'Culmen Length (mm)': 'culmen_length_mm',
        'Culmen Depth (mm)': 'culmen_depth_mm',
        'Flipper Length (mm)': 'flipper_length_mm',
        'Body Mass (g)': 'body_mass_g'
    })
    
    df_filtered = df[['Species', 'Island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'Sex']]
    
    df_filtered.to_csv(output_file, index=False)
    
process_csv('penguins_Iter.csv', 'penguins_new.csv')
