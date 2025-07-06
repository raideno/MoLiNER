import torch
import typing

from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    
    The role of a tokenizer is to convert text strings into token IDs
    and attention masks that can be fed into token encoders.
    """
    
    @abstractmethod
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
        
        Args:
            texts (List[str]): List of text strings to tokenize.
            max_length (Optional[int]): Maximum sequence length. If None, uses model's default.
            padding (bool): Whether to pad sequences to the same length.
            truncation (bool): Whether to truncate sequences that exceed max_length.
            return_tensors (str): Type of tensors to return ("pt" for PyTorch).
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "input_ids": Token IDs tensor of shape (batch_size, seq_len)
                - "attention_mask": Attention mask tensor of shape (batch_size, seq_len)
        """
        pass
    
    @abstractmethod
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
        
        Args:
            text (str): Text string to tokenize.
            max_length (Optional[int]): Maximum sequence length. If None, uses model's default.
            padding (bool): Whether to pad sequence.
            truncation (bool): Whether to truncate sequence that exceeds max_length.
            return_tensors (str): Type of tensors to return ("pt" for PyTorch).
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "input_ids": Token IDs tensor of shape (1, seq_len)
                - "attention_mask": Attention mask tensor of shape (1, seq_len)
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: typing.Union[torch.Tensor, typing.List[int]],
        skip_special_tokens: bool = True,
        batch_index: typing.Optional[int] = None,
    ) -> str:
        """
        Decodes token IDs back to text.
        
        Args:
            token_ids (Union[torch.Tensor, List[int]]): Token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens in output.
            
        Returns:
            str: Decoded text string.
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer."""
        pass
    
    @property
    @abstractmethod
    def model_max_length(self) -> int:
        """Returns the maximum sequence length supported by the model."""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Returns the ID of the padding token."""
        pass
    
    @property
    @abstractmethod
    def cls_token_id(self) -> int:
        """Returns the ID of the classification token."""
        pass
    
    @property
    @abstractmethod
    def sep_token_id(self) -> int:
        """Returns the ID of the separator token."""
        pass