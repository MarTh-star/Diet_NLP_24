import os
import json
import csv

def json_to_csv(json_data, csv_filename):
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")

        # Write the header
        writer.writerow(["Category", "Details", "Source"])

        # Write each category's details and source
        for category, values in json_data.items():
            # Extract the main data and source fields
            details = values.get(category, "")
            source = values.get("source", "")
            
            # Handle cases where source is a list
            if isinstance(source, list):
                source = "; ".join(source)

            # Write the row to the CSV file
            writer.writerow([category, details, source])

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_path = os.path.join(directory, filename)
            csv_filename = os.path.join(directory, filename.replace(".json", ".csv"))

            # Load the JSON data
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            # Convert JSON to CSV
            json_to_csv(json_data, csv_filename)
            print(f"Converted {filename} to {csv_filename}")

# Replace 'your_directory_path' with the path to the directory containing JSON files
directory_path = "Diet_Output_json"
process_directory(directory_path)
