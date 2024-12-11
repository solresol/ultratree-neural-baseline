import argparse
import csv
import sqlite3


def main():
    parser = argparse.ArgumentParser(description='Summarize evaluation results from SQLite databases into a CSV file.')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('databases', nargs='+', help='SQLite database files to process.')
    args = parser.parse_args()

    with open(args.output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['embedding_size', 'hidden_layer_size', 'augmentation', 'number_of_data_points', 'total_loss'])

        for db_path in args.databases:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT description, number_of_data_points, total_loss FROM evaluation_runs;")
            rows = cursor.fetchall()

            for row in rows:
                description, number_of_data_points, total_loss = row
                parts = description.split(',')
                embedding_size = None
                hidden_layer_size = None
                augmentation = None

                for part in parts:
                    if 'Embed=' in part:
                        embedding_size = part.split('=')[1].strip()
                    elif 'Hidden=' in part:
                        hidden_layer_size = part.split('=')[1].strip()
                    else:
                        augmentation = part.strip()

                csv_writer.writerow([embedding_size, hidden_layer_size, augmentation, number_of_data_points, total_loss])

            conn.close()

if __name__ == "__main__":
    main()
