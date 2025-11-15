import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)  # Use module name

class MovieGenresDataset(Dataset):
    """ Dataset class for movie genres dataset.
    It uses a tokenizer to convert text to tokens and a label handler to convert labels.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        tokenizer: Tokenizer from transformers library.
        labelhandler: The label handler to use for labels.
        config (dict): The configuration dictionary.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, labelhandler, config: dict):
        
        self.df = df

        self.tokenizer = tokenizer
        self.labelhandler = labelhandler
        self.config = config
        self.max_len = config.get('tokenizer_max_length', tokenizer.model_max_length)
        logger.info(f"Using max token length: {self.max_len}")
        
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # create a movie description feature by combining title and overview fields
        movie_description = f"Title:{row['title']} Overview: {row['overview']}"

        # tokenizer is guaranteed to be not None
        # we are not allowing truncation here to avoid silent data loss
        text_encoding = self.tokenizer(
            movie_description, truncation=False, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        
        input_tokens = text_encoding['input_ids'].squeeze()
        attention_mask = text_encoding['attention_mask'].squeeze()

        if 'genre_ids' in row:
            # Convert genre_ids to multi-hot encoding
            multi_hot_gt = self.labelhandler.array_to_multi_hot(eval(row['genre_ids']))
        else:
            multi_hot_gt = torch.tensor(torch.nan, dtype=torch.float32).repeat(self.labelhandler.get_multi_hot_length())
        # Combine all features as needed
        features = {'movie_description': movie_description, 'movie_id': row['movie_id'], 'input_tokens': input_tokens, 'attention_mask': attention_mask}

        return features, multi_hot_gt


def create_dataloader(filename, tokenizer, labelhandler, config, fold=None, validation_split:bool=False):
    """ DataLoader factory for the movie genres dataset.
        Dataloader random behavior is determined by torch manual seed from config.
    
    Args:
        filename (str): The path to the CSV file containing the data.
        tokenizer: Tokenizer from transformers library.
        labelhandler: The label handler to use for labels.
        config (dict): The configuration dictionary.
        fold (int): The fold number for cross-validation.
        validation_split (bool): Whether to split the data into training and validation sets.

    Returns:
        DataLoader: A DataLoader for the movie genres dataset.
    """
    df = pd.read_csv(filename)
    logger.info(f"Loaded {len(df)} records from {filename}")
    if config.get('max_dataset_size', None) is not None:
        df = df.head(config['max_dataset_size'])
        logger.info(f"Using only first {config['max_dataset_size']} records from the dataset")
    dataset = MovieGenresDataset(df, tokenizer, labelhandler, config)
    batch_size = config['batch_size']
    validation_fraction = config['validation_fraction']

    if validation_split:
        train_size = int((1 - validation_fraction) * len(df))
        val_size = len(df) - train_size
        seed = config['random_seed']
        generator = torch.Generator().manual_seed(seed)
        # if fold is specified, use it to create a deterministic but shuffled cross validation split
        if fold is not None:
            num_folds = int(np.ceil(1/validation_fraction))
            logger.info(f"Creating fold {fold} of {num_folds} for cross-validation")
            assert fold < num_folds, f"Fold {fold} is out of range for num_folds {num_folds}"
            indices = list(range(len(df)))
            indices = torch.randperm(len(df), generator=generator).tolist()
            fold_size = len(df) // num_folds
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold != num_folds - 1 else len(df)
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        return DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    return DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False), None