# Coffee Quality Institute (CQI) Dataset

## Dataset Source

The Coffee Quality Institute (CQI) dataset is available on Kaggle:
- **Dataset URL**: https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi

## Dataset Description

This dataset contains quality measurements of coffee beans from the Coffee Quality Institute. It includes:

- **Arabica coffee data**: ~1,300 samples with sensory evaluations
- **Robusta coffee data**: ~28 samples with sensory evaluations

### Features

**Sensory Attributes (0-10 scale):**
- Aroma
- Flavor
- Aftertaste
- Acidity
- Body
- Balance
- Uniformity
- Clean Cup
- Sweetness

**Metadata:**
- Country of Origin
- Processing Method
- Variety
- Altitude (meters)
- Harvest Year

**Target Variable:**
- Total Cup Points (0-100 scale)

## Download Instructions

### Option 1: Manual Download
1. Visit https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi
2. Download the dataset (requires Kaggle account)
3. Extract the CSV files to this `data/` folder:
   - `arabica_data_cleaned.csv`
   - `robusta_data_cleaned.csv`

### Option 2: Using Kaggle CLI
```bash
# Install kaggle CLI if not already installed
pip install kaggle

# Configure Kaggle API credentials (requires ~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings -> API -> Create New Token

# Download the dataset
kaggle datasets download -d volpatto/coffee-quality-database-from-cqi -p data/
unzip data/coffee-quality-database-from-cqi.zip -d data/
```

## Files After Download

After downloading, this folder should contain:
- `arabica_data_cleaned.csv` - Arabica coffee quality data
- `robusta_data_cleaned.csv` - Robusta coffee quality data
- `README.md` - This file

## License

Please refer to the Kaggle dataset page for licensing information.
