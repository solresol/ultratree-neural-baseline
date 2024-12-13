import argparse
import csv
import sqlite3
from typing import List


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize evaluation results from SQLite databases into a CSV file.')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('databases', nargs='+', type=List[str], help='SQLite database files to process.')
    args = parser.parse_args()

    with open(args.output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['embedding_size', 'hidden_layer_size', 'augmentation', 'model_parameter_count',  'number_of_data_points', 'total_loss'])

        for db_path in args.databases:  # db_path is str
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # type of db_path is str from args.databases which is List[str]
            cursor.execute("SELECT description, model_parameter_count,  number_of_data_points, total_loss FROM evaluation_runs;")
            rows = cursor.fetchall()

            for row in rows:
                description: str; model_parameter_count: float; number_of_data_points: float; total_loss: float; description: str; model_parameter_count: int; number_of_data_points: int; total_loss: int; description, model_parameter_count,number_of_data_points, total_loss = row
                parts = description.split(',')
                embedding_size: str = None
                hidden_layer_size: str = None
                augmentation: str = None

                for part in parts:
                    if 'Embed=' in part:
                        embedding_size = part.split('=')[1].strip()
                    elif 'Hidden=' in part:
                        hidden_layer_size = part.split('=')[1].strip()
                    else:
                        augmentation = part.strip()

                csv_writer.writerow([embedding_size, hidden_layer_size, augmentation, model_parameter_count, number_of_data_points, total_loss])

            conn.close()

if __name__ == "__main__":
    main()
