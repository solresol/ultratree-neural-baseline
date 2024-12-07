#!/usr/bin/env python

import argparse
import sqlite3
import torch
import torch.nn as nn
import time
import os

# Hyperparameters for model structure
# These should match those used during training
EMBEDDING_DIM = 128
CONTEXT_SIZE = 16
HIDDEN_DIM = 256

class SimpleFFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, output_dim):
        super(SimpleFFNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
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

def load_model(model_file):
    # Load the checkpoint
    checkpoint = torch.load(model_file, map_location='cpu')
    word_sense_to_index = checkpoint['word_sense_to_index']
    index_to_word_sense = {idx: sense for sense, idx in word_sense_to_index.items()}
    vocab_size = len(word_sense_to_index)
    
    model = SimpleFFNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        hidden_dim=HIDDEN_DIM,
        output_dim=vocab_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    model_parameter_count = sum(p.numel() for p in model.parameters())
    
    return model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count

def compute_penalty(correct_path, predicted_path):
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
    # prefix_length could be 0, so penalty = 2^0 = 1 for no match at all.
    # The problem statement: If correct=1.2.3.4.5 and pred=1.2.3.6.7 then prefix=3, penalty=2^-3=0.125
    return 2**(-prefix_length)

def main():
    args = parse_arguments()
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a model on dataset and write inferences.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (checkpoint).')
    parser.add_argument('--input-db', type=str, required=True, help='Path to the input SQLite database (with evaluation data).')
    parser.add_argument('--output-db', type=str, required=True, help='Path to the output SQLite database (for results).')
    parser.add_argument('--description', type=str, required=True, help='Description of this evaluation run.')
    parser.add_argument('--table', type=str, default='training_data', help='Table to read from in input-db (default: training_data).')
    return parser.parse_args()
    # Load model and vocabulary
    model, word_sense_to_index, index_to_word_sense, vocab_size, model_parameter_count = load_model(args.model)
    
    # Connect to input DB and fetch data
    input_conn = sqlite3.connect(args.input_db)
    input_cursor = input_conn.cursor()
    # We assume the table schema includes columns: id, targetword, context1...context16
    query = f"SELECT id, targetword"
    for i in range(1, CONTEXT_SIZE+1):
        query += f", context{i}"
    query += f" FROM {args.table}"
    
    input_cursor.execute(query)
    rows = input_cursor.fetchall()
    input_conn.close()
    
    # Prepare output DB
    output_conn = sqlite3.connect(args.output_db)
def fetch_evaluation_data(args):
    input_conn = sqlite3.connect(args.input_db)
    input_cursor = input_conn.cursor()
    query = f"SELECT id, targetword"
    for i in range(1, CONTEXT_SIZE+1):
        query += f", context{i}"
    query += f" FROM {args.table}"
    
    input_cursor.execute(query)
    rows = input_cursor.fetchall()
    input_conn.close()
    return rows
    
    # Insert a run record (partial)
    # We'll update number_of_data_points and total_loss at the end
    output_cursor.execute("""
        INSERT INTO evaluation_runs(description, model_file, model_parameter_count, context_length, evaluation_datafile, evaluation_table)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (args.description, os.path.abspath(args.model), model_parameter_count, CONTEXT_SIZE, os.path.abspath(args.input_db), args.table))
    
    evaluation_run_id = output_cursor.lastrowid
    output_conn.commit()
    
    # Evaluate
    # We'll iterate through rows, make predictions, compute penalty, and store results
    total_loss = 0.0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for r in rows:
            row_id = r[0]
            correct_path = r[1]
            context_paths = r[2:]
            
            # Map context words to indices
            try:
                context_indices = [word_sense_to_index[cw] for cw in context_paths]
            except KeyError:
                # If there's a sense not in vocab, skip
                # (Shouldn't happen if vocab was built consistently.)
                continue
            
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
    # number_of_data_points = count
    # total_loss = total_loss
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
