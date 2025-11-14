"""This moduele contains functions to train and predict a machine learning model.
Collects inputs, outputs, losses and metrics during training and prediction."""
from kaggle_movie_genres.evaluation import ModelData
import torch.nn as nn
import torch
import logging
import time
from matplotlib import pyplot as plt
import tqdm

logger = logging.getLogger(__name__)

class TrainPredict:
    def __init__(self, name, config, labelhandler, model, train_dataloader, validation_dataloader, test_dataloader):
        self.name = name
        # generate a unique folder name for results with date and time
        self.folder_name = f"results/{self.name}_{time.strftime('%Y%m%d_%H%M%S')}/"
        self.config = config
        self.labelhandler = labelhandler

        self.optimizer = torch.optim.Adam(model.parameters(), float(config['learning_rate']))
        self.device = config['device']
        self.model = model.to(self.device)
        self.compiled_model = torch.compile(self.model)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = nn.BCELoss()

    def _to_device(self, features, labels):
        features = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in features.items()}
        labels = labels.to(self.device)
        return features, labels

    def train_one_epoch(self):
        self.model.train()
        train_data = ModelData(self.config, self.labelhandler)
        for features, labels  in tqdm.tqdm(self.train_dataloader):
            features, labels = self._to_device(features, labels)

            self.optimizer.zero_grad()
            outputs = self.compiled_model(input_ids=features['input_tokens'], attention_mask=features['attention_mask'])
            loss = self.loss_fn(outputs, labels.float())
            loss.backward()
            self.optimizer.step()
            

            train_data.add_batch(features, labels, outputs, loss.item())
            
        
        return train_data.process_epoch_data()
    
    def predict_one_epoch(self, dataloader):
        self.model.eval()
        epoch_data = ModelData(self.config, self.labelhandler)
        with torch.no_grad():
            for features, labels in tqdm.tqdm(dataloader):
                features, labels = self._to_device(features, labels)

                outputs = self.compiled_model(input_ids=features['input_tokens'], attention_mask=features['attention_mask'])
                if not torch.isnan(labels).any():
                    loss = self.loss_fn(outputs, labels.float())
                else:
                    loss = torch.tensor(0.0)
                epoch_data.add_batch(features, labels, outputs, loss.item())
        
        return epoch_data.process_epoch_data()
    
    def train(self):
        self.train_epochs_data = []
        self.val_epochs_data = []
        self.test_epochs_data = []
        for epoch in range(int(self.config['num_epochs'])):
            logger.info(f"Starting epoch {epoch+1}/{self.config['num_epochs']}")
            train_data = self.train_one_epoch()
            logger.info(f"Training epoch {epoch+1} completed. Train F1 Score: {train_data.f1_score:.4f}, Train Loss: {train_data.avg_loss:.4f}")
            val_data = self.predict_one_epoch(self.validation_dataloader)
            logger.info(f"Validation epoch {epoch+1} completed. Val F1 Score: {val_data.f1_score:.4f}, Val Loss: {val_data.avg_loss:.4f}")
            test_data = self.predict_one_epoch(self.test_dataloader)
            logger.info(f"Test epoch {epoch+1} completed. ")
            test_data.submit(self.folder_name, epoch+1, val_data.f1_score)
            self.train_epochs_data.append(train_data)
            self.val_epochs_data.append(val_data)
            self.test_epochs_data.append(test_data)

            logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_data.avg_loss:.4f}, Val Loss: {val_data.avg_loss:.4f}")   
            self._present_results()
                
    def _present_results(self):
        """Create plots of loss, f1 score over epochs for train, validation sets."""
        # f1 score plot
        f1_train = [data.f1_score for data in self.train_epochs_data]
        f1_val = [data.f1_score for data in self.val_epochs_data]
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(f1_train)+1), f1_train, label='Train F1 Score')
        plt.plot(range(1, len(f1_val)+1), f1_val, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.folder_name}/f1_score_epochs.png")
        plt.close()
        # loss plot
        loss_train = [data.losses.mean() for data in self.train_epochs_data]
        loss_val = [data.losses.mean() for data in self.val_epochs_data]
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_train)+1), loss_train, label='Train Loss')
        plt.plot(range(1, len(loss_val)+1), loss_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.folder_name}/loss_epochs.png")
        plt.close()
        