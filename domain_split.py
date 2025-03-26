import os
import json

def extract_video_ids(input_dir, exclude_files, output_dir):
    video_ids_train = []
    video_ids_test = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the directory and process each file
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Process each line in the JSON file
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if 'video_id' in data:
                                    if file in exclude_files:
                                        video_ids_test.append(data['video_id'])
                                    else:
                                        video_ids_train.append(data['video_id'])
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file {file}: {e}")
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # Save the video IDs to output files
    output_train_file = os.path.join(output_dir, 'train_domain.txt')
    output_test_file = os.path.join(output_dir, 'test_domain.txt')

    with open(output_train_file, 'w', encoding='utf-8') as f:
        for video_id in video_ids_train:
            f.write(f"{video_id}\n")

    with open(output_test_file, 'w', encoding='utf-8') as f:
        for video_id in video_ids_test:
            f.write(f"{video_id}\n")

    print(f"Extraction complete. {len(video_ids_train)} video IDs saved to {output_train_file}.")
    print(f"{len(video_ids_test)} video IDs saved to {output_test_file}.")


# List of files to exclude
# exclude_files = ['Education_classified.json', 'Finance_classified.json', 'Military_classified.json']
exclude_files = ['Culture_classified.json']

project_root = os.path.dirname(os.path.abspath(__file__))
input_dir = project_root + '/FakeTT_Domain_output/'
output_dir = project_root + '/FakeTT_Domain_output/split/'

# Call the function
extract_video_ids(input_dir, exclude_files, output_dir)
