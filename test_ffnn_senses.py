import os
import unittest

import torch
from evaluate_model import load_model
from ffnn_senses import SimpleFFNN


class TestModelSaving(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 128
        self.context_size = 16
        self.hidden_dim = 256
        self.output_dim = self.vocab_size
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.model = SimpleFFNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            context_size=self.context_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.temp_file = 'temp_model.pt'

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_model_saving_parameters(self):
        torch.save({
            'epoch': 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'word_sense_to_index': {},
            'embedding_dim': self.embedding_dim,
            'context_size': self.context_size,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }, self.temp_file)

        checkpoint = torch.load(self.temp_file)
        self.assertEqual(checkpoint['embedding_dim'], self.embedding_dim)
        self.assertEqual(checkpoint['context_size'], self.context_size)
        self.assertEqual(checkpoint['hidden_dim'], self.hidden_dim)
        self.assertEqual(checkpoint['batch_size'], self.batch_size)
        self.assertEqual(checkpoint['learning_rate'], self.learning_rate)

    def test_model_loading_parameters(self):
        torch.save({
            'epoch': 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'word_sense_to_index': {str(i): i for i in range(self.vocab_size)},
            'embedding_dim': self.embedding_dim,
            'context_size': self.context_size,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }, self.temp_file)

        model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count = load_model(self.temp_file)
        self.assertEqual(vocab_size, self.vocab_size)
        self.assertEqual(model.embedding.embedding_dim, self.embedding_dim)
        self.assertEqual(model.fc1.in_features, self.context_size * self.embedding_dim)
        self.assertEqual(model.fc1.out_features, self.hidden_dim)
        self.assertEqual(model.fc2.out_features, self.output_dim)

if __name__ == '__main__':
    unittest.main()
class TestFFNNSensesArgs(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(description='Train a simple FFNN on word sense data.')
        self.parser.add_argument('--db-path', type=str, required=True, help='Path to the SQLite database.')
        self.parser.add_argument('--table-name', type=str, default="training_data", help='Name of the table to read from.')
        self.parser.add_argument('--model-save-path', type=str, default='model.pt', help='Path to save or load the model.')
        self.parser.add_argument('--resume', action='store_true', help='Resume training from saved model.')
        self.parser.add_argument('--embedding-dim', type=int, default=128, help='Dimension of the embeddings.')
        self.parser.add_argument('--context-size', type=int, default=16, help='Number of context word senses.')
        self.parser.add_argument('--hidden-dim', type=int, default=256, help='Dimension of the hidden layer.')
        self.parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
        self.parser.add_argument('--num-epochs', type=int, default=1000, help='Maximum number of epochs for training.')
        self.parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer.')

    def test_batch_size_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--batch-size', '512']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.batch_size, 512)

    def test_embedding_dim_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--embedding-dim', '64']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.embedding_dim, 64)

    def test_context_size_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--context-size', '8']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.context_size, 8)

    def test_hidden_dim_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--hidden-dim', '128']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.hidden_dim, 128)

    def test_learning_rate_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--learning-rate', '0.01']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.learning_rate, 0.01)

    def test_num_epochs_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--num-epochs', '500']
        with unittest.mock.patch('sys.argv', test_args):
            args = self.parser.parse_args()
            self.assertEqual(args.num_epochs, 500)

    def test_invalid_batch_size_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--batch-size', '-1']
        with unittest.mock.patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                self.parser.parse_args()

    def test_invalid_learning_rate_argument(self):
        test_args = ['ffnn-senses.py', '--db-path', 'test.db', '--learning-rate', '-0.01']
        with unittest.mock.patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                self.parser.parse_args()
