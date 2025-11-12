""" Create a tokenizer and embedder based on the config """
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)  # Use module name

def create_tokenizer_and_embedder(config):
    model_name = config.get('featurizer_name')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedder = AutoModel.from_pretrained(model_name)
    return tokenizer, embedder