import pytest
from scripts.prepare_data import preprocess_lm


class MockTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Simple mock encoding - just convert each character to a number
        # In real tokenizer this would be more complex
        return [ord(c) % 100 for c in text]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def test_preprocess_lm_basic(mock_tokenizer):
    # Test input
    examples = {"text": ["hello", "world"]}
    max_seq_len = 10

    # Process the examples
    result = preprocess_lm(examples, mock_tokenizer, max_seq_len)

    # Verify the structure of the output
    assert "input_ids" in result
    assert "label_mask" in result
    assert "n_labels" in result

    # Check we have the expected number of sequences
    assert len(result["input_ids"]) == 2
    assert len(result["label_mask"]) == 2
    assert len(result["n_labels"]) == 2

    # Check sequence length
    assert all(len(seq) == max_seq_len for seq in result["input_ids"])
    assert all(len(seq) == max_seq_len for seq in result["label_mask"])

    # Check EOS tokens
    assert result["input_ids"][0][0] == mock_tokenizer.eos_token_id  # Start EOS
    assert not result["label_mask"][0][0]  # Start EOS should not be predicted


def test_preprocess_lm_truncation(mock_tokenizer):
    # Test with a very long input that needs truncation
    long_text = "a" * 100
    examples = {"text": [long_text]}
    max_seq_len = 10

    result = preprocess_lm(examples, mock_tokenizer, max_seq_len)

    # Check that sequences are truncated to max_seq_len
    assert len(result["input_ids"][0]) == max_seq_len
    assert len(result["label_mask"][0]) == max_seq_len


def test_preprocess_lm_padding(mock_tokenizer):
    # Test with a short input that needs padding
    examples = {"text": ["hi"]}
    max_seq_len = 10

    result = preprocess_lm(examples, mock_tokenizer, max_seq_len)

    # Check padding
    assert result["input_ids"][0][-1] == mock_tokenizer.pad_token_id
    assert not result["label_mask"][0][-1]  # Padding should not be predicted
