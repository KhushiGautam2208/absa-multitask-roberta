import torch.nn as nn
from transformers import RobertaModel

class MultiTaskRoBERTa(nn.Module):
    def __init__(self, num_aspect_labels=3, num_sentiment_classes=3, dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(self.roberta.config.hidden_size, num_aspect_labels)
        self.sentiment_head = nn.Linear(self.roberta.config.hidden_size, num_sentiment_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        pooled = outputs.pooler_output
        aspect_logits = self.aspect_head(self.dropout(seq_out))
        sentiment_logits = self.sentiment_head(self.dropout(pooled))
        return aspect_logits, sentiment_logits