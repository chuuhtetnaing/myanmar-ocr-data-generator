import csv
import os
import shutil


def convert_tsv_to_csv(input_file, output_file, headers=None):
    """
    Convert a tab-separated file to CSV format.

    Args:
        input_file (str): Path to the input tab-separated file
        output_file (str): Path to the output CSV file
        headers (list, optional): Column headers to add to the CSV file
    """
    # Read the tab-separated file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # No need to create directories as they already exist

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write headers if provided
        if headers:
            writer.writerow(headers)

        # Write data rows
        for line in lines:
            # Split by tab and strip whitespace
            fields = [field.strip() for field in line.split('\t')]
            fields[0] = fields[0].split('/', 1)[1]
            writer.writerow(fields)

    print(f"Converted {input_file} to {output_file}")


# Example usage
if __name__ == "__main__":
    # Folders already exist, so no need to create them

    # Convert train.txt to train.csv with headers
    convert_tsv_to_csv(
        'dataset/train.txt',
        'dataset/train/metadata.csv',
        headers=['file_name', 'text']
    )

    # Convert test.txt to test.csv with headers
    convert_tsv_to_csv(
        'dataset/test.txt',
        'dataset/test/metadata.csv',
        headers=['file_name', 'text']
    )

    print("CSV files have been copied to train and test folders.")