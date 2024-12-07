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
