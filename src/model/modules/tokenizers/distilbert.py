import torch
import typing

from transformers import AutoTokenizer

from .index import BaseTokenizer

class DistilBertTokenizer(BaseTokenizer):
    """
    DistilBERT tokenizer implementation using HuggingFace's transformers library.
    
    This tokenizer wraps the AutoTokenizer from HuggingFace to provide a consistent
    interface for tokenizing text, specifically for use with the TMR pretrained model.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initializes the DistilBERT tokenizer.
        
        Args:
            model_name (str): The name of the DistilBERT model to load the tokenizer for.
                            Defaults to "distilbert-base-uncased".
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize(
        self, 
        texts: typing.List[str],
        max_length: typing.Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        batch_index: typing.Optional[int] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Tokenizes a list of text strings.
        """
        if max_length is None:
            max_length = self.model_max_length
            
        result = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return result
    
    def encode(
        self,
        text: str,
        max_length: typing.Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        batch_index: typing.Optional[int] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Tokenizes a single text string.
        """
        if max_length is None:
            max_length = self.model_max_length
            
        result = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return result
    
    def decode(
        self,
        token_ids: typing.Union[torch.Tensor, typing.List[int]],
        skip_special_tokens: bool = True,
        batch_index: typing.Optional[int] = None,
    ) -> str:
        """
        Decodes token IDs back to text.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer."""
        return len(self.tokenizer)
    
    @property
    def model_max_length(self) -> int:
        """Returns the maximum sequence length supported by the model."""
        return self.tokenizer.model_max_length
    
    @property
    def pad_token_id(self) -> int:
        """Returns the ID of the padding token."""
        return self.tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        """Returns the ID of the classification token."""
        return self.tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        """Returns the ID of the separator token."""
        return self.tokenizer.sep_token_id
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: typing.Union[
            typing.List[str],
            typing.List[typing.Tuple[str, str]],
            typing.List[typing.List[str]]
        ],
        max_length: typing.Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        batch_index: typing.Optional[int] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Batch tokenization for multiple texts or text pairs.
        
        This method is useful for batch processing of prompts in the model.
        """
        if max_length is None:
            max_length = self.model_max_length
            
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
