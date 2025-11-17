from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

class CLS_Classifier(nn.Module):
    def __init__(self, bert_embedder: AutoModel, num_labels, config: dict):
        super().__init__()
        self.config = config
        self.bert_embedder = bert_embedder
        for param in self.bert_embedder.parameters():
            param.requires_grad = False
        self.latent_layer = nn.Linear(self.bert_embedder.config.hidden_size, self.bert_embedder.config.hidden_size*config['classifier_hidden_size_factor'])
        self.classifier = nn.Linear(self.bert_embedder.config.hidden_size*config['classifier_hidden_size_factor'], num_labels)
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_embedder(input_ids, attention_mask=attention_mask)
        if self.config['classifier_type'] == 'Pool_Classifier':
            embedding = outputs.last_hidden_state.mean(dim=1)
        else:  # CLS_Classifier
            embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        latent_output = nn.ReLU()(self.latent_layer(self.dropout(embedding)))
        
        logits = self.classifier(self.dropout(latent_output))
        probabilities = torch.sigmoid(logits)
        return probabilities
