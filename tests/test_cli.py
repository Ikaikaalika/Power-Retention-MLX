import pytest
import sys
import os
from unittest.mock import patch

# Add root to path to import train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import parse_args

def test_parse_args_defaults():
    with patch.object(sys, 'argv', ['train.py']):
        args = parse_args()
        assert args.mode == 'llm'
        assert args.power == 2
        assert args.model == 'tiny'

def test_parse_args_power1():
    with patch.object(sys, 'argv', ['train.py', '--power', '1']):
        args = parse_args()
        assert args.power == 1

def test_parse_args_layer_mode():
    with patch.object(sys, 'argv', ['train.py', '--mode', 'layer', '--dim', '128']):
        args = parse_args()
        assert args.mode == 'layer'
        assert args.dim == 128
