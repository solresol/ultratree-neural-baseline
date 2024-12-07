import unittest
from unittest.mock import MagicMock, patch

import torch
from ffnn-senses import argparse
from ffnn_senses import (SimpleFFNN, WordSenseDataset, build_word_sense_vocab,
                         main)
from torch.utils.data import DataLoader


class TestSimpleFFNN(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 128
        self.context_size = 16
        self.hidden_dim = 256
        self.output_dim = 100
        self.model = SimpleFFNN(self.vocab_size, self.embedding_dim, self.context_size, self.hidden_dim, self.output_dim)

    def test_forward(self):
        context_words = torch.randint(0, self.vocab_size, (32, self.context_size))
        output = self.model(context_words)
        self.assertEqual(output.shape, (32, self.output_dim))
        self.assertIsInstance(output, torch.Tensor)

class TestMainFunction(unittest.TestCase):
    @patch('ffnn_senses.sqlite3.connect')
    @patch('ffnn_senses.build_word_sense_vocab')
    @patch('ffnn_senses.WordSenseDataset')
    @patch('ffnn_senses.SimpleFFNN')
    def test_main(self, MockSimpleFFNN, MockWordSenseDataset, mock_build_vocab, mock_sqlite_connect):
        mock_build_vocab.return_value = ({'sense1': 0, 'sense2': 1}, {0: 'sense1', 1: 'sense2'}, 2)
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        MockWordSenseDataset.return_value = mock_dataset

        with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            db_path='test.db', table_name='training_data', model_save_path='model.pt', resume=False)):
            main()

        MockSimpleFFNN.assert_called_once_with(vocab_size=2, embedding_dim=128, context_size=16, hidden_dim=256, output_dim=2)
        MockWordSenseDataset.assert_called_once_with('test.db', 'training_data', {'sense1': 0, 'sense2': 1})
        self.assertEqual(mock_dataset.__len__.call_count, 1)

if __name__ == '__main__':
    unittest.main()
