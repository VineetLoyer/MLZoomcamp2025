# MBTI Dataset Download Instructions

## Dataset Information

The MBTI (Myers-Briggs Type Indicator) dataset contains posts from users on the PersonalityCafe forum, along with their self-reported MBTI personality types.

**Dataset Details:**
- **Source**: Kaggle
- **File**: `mbti_1.csv`
- **Size**: ~10MB
- **Records**: ~8,600 users
- **Columns**:
  - `type`: MBTI personality type (e.g., INTJ, ENFP)
  - `posts`: Last 50 posts from the user, separated by `|||`

## Download Instructions

### Option 1: Using Kaggle Website

1. Go to the dataset page: https://www.kaggle.com/datasets/datasnaek/mbti-type
2. Click the "Download" button (you'll need a Kaggle account)
3. Extract the downloaded zip file
4. Copy `mbti_1.csv` to this `data/` folder

### Option 2: Using Kaggle CLI

If you have the Kaggle CLI installed and configured:

```bash
# Install kaggle CLI if not already installed
pip install kaggle

# Download the dataset
kaggle datasets download -d datasnaek/mbti-type

# Extract the zip file
unzip mbti-type.zip -d .

# Clean up
rm mbti-type.zip
```

### Option 3: Using Kaggle API in Python

```python
import kaggle

# Download dataset
kaggle.api.dataset_download_files('datasnaek/mbti-type', path='data/', unzip=True)
```

## Setting Up Kaggle Credentials

To use the Kaggle CLI or API:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to your account settings: https://www.kaggle.com/settings
3. Scroll to "API" section and click "Create New Token"
4. This downloads `kaggle.json`
5. Place the file in:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

## Expected File Structure

After downloading, your `data/` folder should contain:

```
data/
├── mbti_1.csv      # The main dataset file
└── README.md       # This file
```

## Data Preview

The dataset looks like this:

| type | posts |
|------|-------|
| INFJ | 'http://www.youtube.com/watch...\|\|\|I'm finding...\|\|\|...' |
| ENTP | 'I'm going to...\|\|\|What do you think...\|\|\|...' |
| ... | ... |

## Notes

- The dataset is provided for educational purposes
- Each user has approximately 50 posts concatenated with `|||` delimiter
- Some posts may contain URLs, special characters, and MBTI type mentions that need preprocessing
