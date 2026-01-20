"""Data Loader module for Coffee Quality Prediction."""

import pandas as pd
from typing import List, Optional


class DataLoader:
    """Loads and merges coffee quality datasets."""
    
    COMMON_COLUMNS = [
        'Species', 'Owner', 'Country.of.Origin', 'Farm.Name', 'Lot.Number',
        'Mill', 'ICO.Number', 'Company', 'Altitude', 'Region', 'Producer',
        'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner', 'Harvest.Year',
        'Grading.Date', 'Owner.1', 'Variety', 'Processing.Method',
        'Flavor', 'Aftertaste', 'Clean.Cup', 'Cupper.Points',
        'Total.Cup.Points', 'Moisture', 'Category.One.Defects', 'Quakers',
        'Color', 'Category.Two.Defects', 'Expiration', 'Certification.Body',
        'Certification.Address', 'Certification.Contact', 'unit_of_measurement',
        'altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters'
    ]
    
    ROBUSTA_COLUMN_MAPPING = {
        'Fragrance...Aroma': 'Aroma',
        'Salt...Acid': 'Acidity',
        'Bitter...Sweet': 'Sweetness',
        'Mouthfeel': 'Body',
        'Uniform.Cup': 'Uniformity'
    }
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load coffee dataset from CSV file."""
        try:
            df = pd.read_csv(filepath)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Dataset file is empty: {filepath}")
        except pd.errors.ParserError:
            raise ValueError(f"Invalid CSV format: {filepath}")
    
    def merge_datasets(
        self,
        arabica_df: pd.DataFrame,
        robusta_df: pd.DataFrame,
        use_common_columns: bool = True
    ) -> pd.DataFrame:
        """Merge Arabica and Robusta datasets with common columns."""
        arabica = arabica_df.copy()
        robusta = robusta_df.copy()
        
        robusta = robusta.rename(columns=self.ROBUSTA_COLUMN_MAPPING)
        
        if use_common_columns:
            arabica_cols = set(arabica.columns)
            robusta_cols = set(robusta.columns)
            common_cols = list(arabica_cols.intersection(robusta_cols))
            arabica = arabica[common_cols]
            robusta = robusta[common_cols]
        
        merged_df = pd.concat([arabica, robusta], ignore_index=True)
        return merged_df
    
    def load_and_merge(
        self,
        arabica_path: str,
        robusta_path: str,
        use_common_columns: bool = True
    ) -> pd.DataFrame:
        """Load and merge both datasets in one call."""
        arabica_df = self.load_data(arabica_path)
        robusta_df = self.load_data(robusta_path)
        return self.merge_datasets(arabica_df, robusta_df, use_common_columns)
