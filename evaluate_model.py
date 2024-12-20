#!/usr/bin/env python

import argparse
import sqlite3
import torch
import torch.nn as nn
import time
from typing import Dict, Tuple, List
import os
import sys

OOV_TOKEN = "<OOV>"

class SimpleFFNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int, hidden_dim: int, output_dim: int) -> None:
        super(SimpleFFNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Use index 0 for OOV token
        )
        self.fc1 = nn.Linear(in_features=context_size * embedding_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, context_words):
        embeddings = self.embedding(context_words)  # (batch_size, context_size, embedding_dim)
        embeddings = embeddings.view(embeddings.size(0), -1)  # (batch_size, context_size*embedding_dim)
        out = self.fc1(embeddings)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def load_model(model_file: str) -> Tuple[SimpleFFNN, Dict[str, int], Dict[int, str], int, int, int]:
    """
    Loads a neural network model from a checkpoint file.

    Args:
        model_file (str): Path to the model checkpoint file.

    Returns:
        model (SimpleFFNN): The loaded neural network model.
        word_sense_to_index (dict): Mapping from word senses to their indices.
        index_to_word_sense (dict): Mapping from indices to word senses.
        vocab_size (int): Size of the vocabulary.
        model_parameter_count (int): Total number of parameters in the model.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    word_sense_to_index = checkpoint['word_sense_to_index']
    index_to_word_sense = {idx: sense for sense, idx in word_sense_to_index.items()}
    vocab_size = len(word_sense_to_index)
    context_size = checkpoint['context_size']
    
    model = SimpleFFNN(
        vocab_size=vocab_size,
        embedding_dim=checkpoint['embedding_dim'],
        context_size=checkpoint['context_size'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=vocab_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    model_parameter_count = sum(p.numel() for p in model.parameters())
    
    return model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count, context_size

def compute_penalty(correct_path: str, predicted_path: str) -> float:
    """
    Computes the penalty based on the similarity between the correct and predicted paths.

    Args:
        correct_path (str): The correct path as a string.
        predicted_path (str): The predicted path as a string.

    Returns:
        float: The penalty value, where 0.0 indicates an exact match and higher values indicate greater dissimilarity.
    """
    if correct_path == predicted_path:
        # Exact match
        return 0.0
    correct_parts = correct_path.split('.')
    predicted_parts = predicted_path.split('.')
    prefix_length = 0
    for c, p in zip(correct_parts, predicted_parts):
        if c == p:
            prefix_length += 1
        else:
            break
    # If not exact match, penalty = 2^(-prefix_length)
    return 2**(-prefix_length)

def main() -> None:
    """
    Main function to evaluate the model.

    This function parses command-line arguments, loads the model, connects to the input and output databases,
    fetches evaluation data, performs predictions, computes penalties, and stores the results in the database.
    """
    parser = argparse.ArgumentParser(description='Evaluate a model on dataset and write inferences.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (checkpoint).')
    parser.add_argument('--input-db', type=str, required=True, help='Path to the input SQLite database (with evaluation data).')
    parser.add_argument('--output-db', type=str, required=True, help='Path to the output SQLite database (for results).')
    parser.add_argument('--description', type=str, required=True, help='Description of this evaluation run.')
    parser.add_argument('--table', type=str, default='training_data', help='Table to read from in input-db (default: training_data).')
    args = parser.parse_args()
    
    # Load model and vocabulary
    model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count, context_size = load_model(args.model)
    
    # Connect to input DB and fetch data
    input_conn = sqlite3.connect(args.input_db)
    input_cursor = input_conn.cursor()
    # We assume the table schema includes columns: id, targetword, context1...context16
    query = f"SELECT id, targetword"
    for i in range(1, context_size+1):
        query += f", context{i}"
    query += f" FROM {args.table}"
    
    input_cursor.execute(query)
    rows = input_cursor.fetchall()
    input_conn.close()
    
    # Prepare output DB
    output_conn = sqlite3.connect(args.output_db)
    output_cursor = output_conn.cursor()
    
    # Create tables if not exist
    output_cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            evaluation_run_id integer primary key,
            evaluation_start_time timestamp default current_timestamp,
            evaluation_end_time timestamp,
            description text not null,
            model_file text not null,
            model_parameter_count integer,
            context_length integer,
            evaluation_datafile text not null,
            evaluation_table text not null,
            number_of_data_points integer,
            total_loss float
        )
    """)
    output_cursor.execute("""
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY,
            validation_run_id integer references evaluation_runs(evaluation_run_id),
            input_id INTEGER,
            predicted_path TEXT,
            correct_path TEXT,
            loss REAL,
            when_predicted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert a run record (partial)
    output_cursor.execute("""
        INSERT INTO evaluation_runs(description, model_file, model_parameter_count, context_length, evaluation_datafile, evaluation_table)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (args.description, os.path.abspath(args.model), model_parameter_count, context_size, os.path.abspath(args.input_db), args.table))
    
    evaluation_run_id = output_cursor.lastrowid
    output_conn.commit()
    
    # Evaluate
    total_loss = 0.0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for r in rows:
            row_id = r[0]
            correct_path = r[1]
            context_paths = r[2:]
            
            # Map context words to indices, using OOV_INDEX for out-of-vocabulary terms
            context_indices = [word_sense_to_index.get(cw, 0) for cw in context_paths]
            
            context_tensor = torch.tensor([context_indices], dtype=torch.long)
            outputs = model(context_tensor)  # (1, vocab_size)
            pred_idx = torch.argmax(outputs, dim=1).item()
            predicted_path = index_to_word_sense[pred_idx]
            
            loss = compute_penalty(correct_path, predicted_path)
            total_loss += loss
            count += 1
            
            # Insert into inferences
            output_cursor.execute("""
                INSERT INTO inferences(validation_run_id, input_id, predicted_path, correct_path, loss)
                VALUES (?, ?, ?, ?, ?)
            """, (evaluation_run_id, row_id, predicted_path, correct_path, loss))
    
    # Update evaluation_runs with final info
    output_cursor.execute("""
        UPDATE evaluation_runs
        SET evaluation_end_time = current_timestamp,
            number_of_data_points = ?,
            total_loss = ?
        WHERE evaluation_run_id = ?
    """, (count, total_loss, evaluation_run_id))
    
    output_conn.commit()
    output_conn.close()

if __name__ == "__main__":
    main()
