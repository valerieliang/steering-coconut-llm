# steering/core/__init__.py
from .model_utils import load_model, load_tokenizer, get_thought_vector, build_input
from .contrastive_pairs import build_swap_pairs, build_cross_pairs, build_pairs
from .steering_injection import create_steering_forward, generate_with_steering, extract_answer_word, get_logit_margin

__all__ = [
    "load_model", 
    "load_tokenizer", 
    "get_thought_vector", 
    "build_input",
    "build_swap_pairs", 
    "build_cross_pairs", 
    "build_pairs",
    "create_steering_forward",
    "generate_with_steering", 
    "extract_answer_word", 
    "get_logit_margin"
]