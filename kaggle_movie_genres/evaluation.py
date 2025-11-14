"""A class to hold input, output, loss and metrics during training and prediction.
Also performs evaluation of the model."""
import torch
from sklearn.metrics import f1_score
import numpy as np

class ModelData:
    def __init__(self, config, labelhandler):
        self.config = config
        self.labelhandler = labelhandler
        self.movie_ids = []
        self.movie_descriptions = []
        self.true_labels = []
        self.predicted_labels = []
        self.probs = []
        self.losses = []
        self.metrics = []
        
    def add_batch(self, features, true_labels, probs, loss):
        self.movie_ids.append(features['movie_id'].cpu())
        self.probs.append(probs.detach().cpu())
        self.movie_descriptions.append(features['movie_description'])
        self.true_labels.append(true_labels.cpu())
        self.losses.append(loss)
            
    
    def process_epoch_data(self):
        self.movie_ids = torch.cat(self.movie_ids).numpy()
        self.probs = torch.cat(self.probs).numpy()
        self.true_labels = torch.cat(self.true_labels).numpy()
        self.losses = torch.tensor(self.losses).numpy()
        
        self.predicted_labels = (self.probs >= self.config.get('classification_threshold', 0.5)).astype(int)
        self.predicted_label_arrays = [self.labelhandler.multi_hot_to_array(pred) for pred in self.predicted_labels]
        self.predicted_label_names = [self.labelhandler.array_to_label_names(arr) for arr in self.predicted_label_arrays]
        if np.isnan(self.true_labels).any():
            self.f1_score = float('nan')
        else:
            self.f1_score = f1_score(self.true_labels, self.predicted_labels, average='macro')
        self.avg_loss = np.mean(self.losses)
        return self
        
    def submit(self, folder_name, epoch, val_f1_score):
        from kaggle_movie_genres.submission import format_predictions
        import os
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        filename = os.path.join(folder_name, f"submission_epoch_{epoch}_valF1_{val_f1_score*1000:.0f}.csv")
        format_predictions(filename, self.movie_ids, self.predicted_labels, self.labelhandler)
        # write debug info
        debug_filename = os.path.join(folder_name, f"debug_epoch_{epoch}_valF1_{val_f1_score*1000:.0f}.csv")
        with open(debug_filename, 'w', encoding='utf-8') as f:
            f.write("movie_id,movie_description,true_genres,predicted_genres,probabilities\n")
            for movie_id, description, true_labels, pred_labels, probs in zip(
                self.movie_ids, self.movie_descriptions, self.true_labels, self.predicted_label_arrays, self.probs
            ):
                true_genre_names = self.labelhandler.array_to_label_names(self.labelhandler.multi_hot_to_array(true_labels))
                pred_genre_names = self.labelhandler.array_to_label_names(pred_labels)
                prob_str = ' '.join([f"{p:.4f}" for p in probs])
                f.write(f"{movie_id},\"{description}\",\"{' '.join(true_genre_names)}\",\"{' '.join(pred_genre_names)}\",\"{prob_str}\"\n")