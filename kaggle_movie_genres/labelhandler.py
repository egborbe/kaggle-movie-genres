"""A module for handling labels in the dataset.
It gets the location of the label file from the config file.
parses the label file and provides methods to access label information.
can convert from array to multi-hot encoding and vice versa."""
import pandas as pd
import numpy as np

class LabelHandler:
    def __init__(self, config):
        self.config = config
        self.label_info = self._load_label_info()

    def _load_label_info(self):
        label_file = self.config.get('label_file')
        df = pd.read_csv(label_file)
        return df

    def array_to_multi_hot(self, label: list[int]):
        """Convert a single array label to multi-hot encoding.
        
        Args:
            label (list[int]): The label to convert.
        Returns:
            np.ndarray: The multi-hot encoding.
        """
        multi_hot = np.zeros(len(self.label_info), dtype=np.float32)
        for id in label:
            row_index = self.label_info[self.label_info['id'] == id].index
            assert len(row_index) == 1, f"Expected single index for id {id}, got {len(row_index)}"
            multi_hot[row_index[0]] = 1
        return multi_hot

    def multi_hot_to_array(self, multi_hot: np.ndarray):
        """Convert multi-hot encoding to array label.
        
        Args:
            multi_hot (np.ndarray): The multi-hot encoding to convert.
        Returns:
            list[int]: The array label.
        """
        return self.label_info['id'][multi_hot.astype(bool)].values

    def array_to_label_names(self, labels: list[int]):
        """Convert array label to label names.
        Args:
            labels (list[int]): The label to convert.
        Returns:
            list[str]: The label names.
        """
        label_names = []
        for label in labels:
            name = self.label_info[self.label_info['id'] == label]['name'].values[0]
            label_names.append(name)
        return label_names

    def get_multi_hot_length(self):
        """Get the length of the multi-hot encoding.
        
        Returns:
            int: The length of the multi-hot encoding.
        """
        return len(self.label_info)
