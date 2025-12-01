import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import quote

def search_wine_grape(wine_name, winery, region):
    """
    Search for grape variety information using multiple strategies
    """
    # Clean the wine name
    wine_name_clean = re.sub(r'\d{4}$', '', wine_name).strip()
    
    # Common grape varieties to look for in wine names
    grape_keywords = {
        'Cabernet Sauvignon': ['Cabernet Sauvignon', 'Cabernet'],
        'Merlot': ['Merlot'],
        'Pinot Noir': ['Pinot Noir'],
        'Syrah/Shiraz': ['Syrah', 'Shiraz'],
        'Tempranillo': ['Tempranillo'],
        'Sangiovese': ['Sangiovese', 'Chianti', 'Brunello'],
        'Nebbiolo': ['Nebbiolo', 'Barolo', 'Barbaresco'],
        'Grenache': ['Grenache', 'Garnacha'],
        'Zinfandel': ['Zinfandel'],
        'Primitivo': ['Primitivo'],
        'Malbec': ['Malbec'],
        'Carmenere': ['Carmenere'],
        'Petit Verdot': ['Petit Verdot'],
        'Corvina': ['Corvina', 'Valpolicella', 'Amarone'],
        'Montepulciano': ['Montepulciano'],
        'Barbera': ['Barbera'],
        'Dolcetto': ['Dolcetto'],
        'Nero d\'Avola': ['Nero d\'Avola'],
        'Aglianico': ['Aglianico'],
        'Touriga Nacional': ['Touriga Nacional'],
        'Pinotage': ['Pinotage'],
    }
    
    # Check wine name for grape variety
    wine_name_lower = wine_name_clean.lower()
    region_lower = region.lower() if region else ''
    
    for grape, keywords in grape_keywords.items():
        for keyword in keywords:
            if keyword.lower() in wine_name_lower or keyword.lower() in region_lower:
                return grape
    
    # Special region-based mapping
    region_grape_map = {
        'Pomerol': 'Merlot',
        'Saint-Émilion': 'Merlot, Cabernet Franc',
        'Margaux': 'Cabernet Sauvignon',
        'Pauillac': 'Cabernet Sauvignon',
        'Priorat': 'Grenache, Carignan',
        'Ribera del Duero': 'Tempranillo',
        'Rioja': 'Tempranillo',
        'Bolgheri': 'Cabernet Sauvignon, Merlot',
        'Toscana': 'Sangiovese',
        'Gigondas': 'Grenache',
        'Châteauneuf-du-Pape': 'Grenache',
        'Napa Valley': 'Cabernet Sauvignon',
        'Sonoma': 'Pinot Noir',
        'Hermitage': 'Syrah',
        'Côte-Rôtie': 'Syrah',
    }
    
    for region_key, grape in region_grape_map.items():
        if region_key.lower() in region_lower:
            return grape
    
    return 'Unknown'

def add_grape_column_to_csv():
    """
    Read Red.csv, add Grape column, and save
    """
    print("Loading Red.csv...")
    df = pd.read_csv('Red.csv')
    
    print(f"Processing {len(df)} wines...")
    
    grapes = []
    for idx, row in df.iterrows():
        wine_name = row['Name']
        winery = row['Winery']
        region = row['Region']
        
        grape = search_wine_grape(wine_name, winery, region)
        grapes.append(grape)
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(df)} wines...")
    
    # Add the Grape column
    df['Grape'] = grapes
    
    # Save back to CSV
    df.to_csv('Red.csv', index=False)
    
    print(f"\nCompleted! Added Grape column to Red.csv")
    print(f"\nGrape variety distribution:")
    print(df['Grape'].value_counts())
    
    # Show some examples
    print("\nSample results:")
    print(df[['Name', 'Region', 'Grape']].head(20).to_string(index=False))

if __name__ == "__main__":
    add_grape_column_to_csv()
