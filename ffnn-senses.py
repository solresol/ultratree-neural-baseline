#!/usr/bin/env python3

import argparse
import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Define hyperparameters
EMBEDDING_DIM = 128     # Dimension of the embeddings
CONTEXT_SIZE = 16       # Number of context word senses
HIDDEN_DIM = 256        # Dimension of the hidden layer
BATCH_SIZE = 1024       # Batch size for training
NUM_EPOCHS = 1000       # Maximum number of epochs for training
LEARNING_RATE = 0.001   # Learning rate for optimizer

class WordSenseDataset(Dataset):
    def __init__(self, db_path, table_name, word_sense_to_index):
        self.db_path = db_path
        self.table_name = table_name
        self.word_sense_to_index = word_sense_to_index
        self.data = self.load_data()
        
    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT targetword, context1, context2, context3, context4, context5, context6, context7, context8, context9, context10, context11, context12, context13, context14, context15, context16 FROM {self.table_name}")
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target_sense = sample[0]
        context_senses = sample[1:]
        
        # Map word senses to indices
        target_index = self.word_sense_to_index[target_sense]
        context_indices = [self.word_sense_to_index[sense] for sense in context_senses]
        
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_index, dtype=torch.long)
        return context_tensor, target_tensor

class SimpleFFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, output_dim):
        super(SimpleFFNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # The input dimension is context_size * embedding_dim since embeddings are concatenated
        self.fc1 = nn.Linear(in_features=context_size * embedding_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, context_words):
        """
        context_words: Tensor of shape (batch_size, context_size)
        """
        # Get embeddings for each word in the context
        embeddings = self.embedding(context_words)  # Shape: (batch_size, context_size, embedding_dim)
        
        # Flatten the embeddings to concatenate them
        embeddings = embeddings.view(embeddings.size(0), -1)  # Shape: (batch_size, context_size * embedding_dim)
        
        # Forward pass through the network
        out = self.fc1(embeddings)
        out = self.relu(out)
        out = self.fc2(out)  # Shape: (batch_size, output_dim)
        
        return out

def build_word_sense_vocab(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT targetword FROM {table_name}")
    target_senses = set(row[0] for row in cursor.fetchall())
    
    # Get unique context senses
    context_senses = set()
    for i in range(1, CONTEXT_SIZE + 1):
        cursor.execute(f"SELECT DISTINCT context{i} FROM {table_name}")
        context_senses.update(row[0] for row in cursor.fetchall())
    
    conn.close()
    
    # Combine target senses and context senses
    all_senses = target_senses.union(context_senses)
    
    # Create mappings
    word_sense_to_index = {sense: idx for idx, sense in enumerate(sorted(all_senses))}
    index_to_word_sense = {idx: sense for sense, idx in word_sense_to_index.items()}
    
    vocab_size = len(word_sense_to_index)
    return word_sense_to_index, index_to_word_sense, vocab_size

def main():
    parser = argparse.ArgumentParser(description='Train a simple FFNN on word sense data.')
    parser.add_argument('--db-path', type=str, required=True, help='Path to the SQLite database.')
    parser.add_argument('--table-name', type=str, default="training_data", help='Name of the table to read from.')
    parser.add_argument('--model-save-path', type=str, default='model.pt', help='Path to save or load the model.')
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model.')
    args = parser.parse_args()
    
    # Build vocabulary mappings
    print("Building vocabulary...")
    word_sense_to_index, index_to_word_sense, vocab_size = build_word_sense_vocab(args.db_path, args.table_name)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset and dataloader
    dataset = WordSenseDataset(args.db_path, args.table_name, word_sense_to_index)
    # Define the split sizes
    validation_split = 0.1  # 10% of the data for validation
    dataset_size = len(dataset)
    validation_size = int(validation_split * dataset_size)
    training_size = dataset_size - validation_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [training_size, validation_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Instantiate the model
    model = SimpleFFNN(
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM, help='Dimension of the embeddings.')
    parser.add_argument('--context-size', type=int, default=CONTEXT_SIZE, help='Number of context word senses.')
    parser.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM, help='Dimension of the hidden layer.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS, help='Maximum number of epochs for training.')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate for optimizer.')
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        context_size=args.context_size,
        hidden_dim=args.hidden_dim,
        output_dim=vocab_size  # Output dimension is the same as vocabulary size
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load model state if resuming
    if args.resume and os.path.exists(args.model_save_path):
        print("Loading model state from", args.model_save_path)
        checkpoint = torch.load(args.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        word_sense_to_index = checkpoint['word_sense_to_index']
    else:
        start_epoch = 1

    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait before stopping
    patience_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (context_batch, target_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(context_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] completed. Average Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context_batch, target_batch in val_loader:
                outputs = model(context_batch)
                loss = criterion(outputs, target_batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
    
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] completed. Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")
    
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'word_sense_to_index': word_sense_to_index,
                'embedding_dim': EMBEDDING_DIM,
                'context_size': CONTEXT_SIZE,
                'hidden_dim': HIDDEN_DIM,
                'batch_size': BATCH_SIZE,
                'learning_rate': args.learning_rate
            }, args.model_save_path)
            print(f"Validation loss improved. Model saved to {args.model_save_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
if __name__ == "__main__":
    main()
