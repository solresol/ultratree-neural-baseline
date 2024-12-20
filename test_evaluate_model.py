import sqlite3
import unittest
from unittest.mock import MagicMock, patch

import torch
from evaluate_model import compute_penalty, load_model
from evaluate_model import main as evaluate_main


class TestEvaluateModel(unittest.TestCase):

    @patch('evaluate_model.torch.load')
    def test_load_model(self, mock_torch_load):
        mock_checkpoint = {
            'word_sense_to_index': {'sense1': 0, 'sense2': 1},
            'model_state_dict': MagicMock()
        }
        mock_torch_load.return_value = mock_checkpoint

        model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count = load_model('dummy_model_path')

        self.assertEqual(word_sense_to_index, {'sense1': 0, 'sense2': 1})
        self.assertEqual(index_to_word_sense, {0: 'sense1', 1: 'sense2'})
        self.assertEqual(vocab_size, 2)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertGreater(model_parameter_count, 0)

    def test_compute_penalty(self):
        self.assertEqual(compute_penalty('1.2.3', '1.2.3'), 0.0)
        self.assertEqual(compute_penalty('1.2.3', '1.2.4'), 0.5)
        self.assertEqual(compute_penalty('1.2.3', '1.3.4'), 1.0)
        self.assertEqual(compute_penalty('1.2.3.4', '1.2.3.5'), 0.125)
        self.assertEqual(compute_penalty('1.2.3', '2.3.4'), 1.0)

    def test_oov_handling(self):
        word_sense_to_index = {'sense1': 0, 'sense2': 1}
        context_paths = ['sense1', 'unknown_sense']
        OOV_INDEX = -1
        context_indices = [word_sense_to_index.get(cw, OOV_INDEX) for cw in context_paths]
        self.assertEqual(context_indices, [0, OOV_INDEX])
        self.assertEqual(compute_penalty('1.2.3', '2.3.4'), 1.0)

    @patch('evaluate_model.sqlite3.connect')
    @patch('evaluate_model.torch.load')
    def test_main_evaluation_logic(self, mock_torch_load, mock_sqlite_connect):
        mock_checkpoint = {
            'word_sense_to_index': {'context1': 0, 'context2': 1},
            'model_state_dict': MagicMock()
        }
        mock_torch_load.return_value = mock_checkpoint

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            (1, '1.2.3', 'context1', 'context2'),
            (2, '1.2.4', 'context2', 'context1')
        ]

        with patch('evaluate_model.SimpleFFNN.forward', return_value=torch.tensor([[0.1, 0.9]])):
            evaluate_main()

        self.assertEqual(mock_cursor.execute.call_count, 4)
        self.assertEqual(mock_cursor.execute.call_args_list[2][0][0].startswith('INSERT INTO inferences'), True)
        self.assertEqual(mock_cursor.execute.call_args_list[3][0][0].startswith('UPDATE evaluation_runs'), True)

if __name__ == '__main__':
    unittest.main()
