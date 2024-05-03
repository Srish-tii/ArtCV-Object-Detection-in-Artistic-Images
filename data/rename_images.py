import os
import shutil

def rename_images(source_folder, target_folder):
    # Dictionary to hold target names mapping
    target_name_map = {}

    # Read target folder filenames and create a mapping
    for filename in os.listdir(target_folder):
        base_name = filename.split('-')[0]
        target_name_map[base_name] = filename
    
    # Rename files in the source folder based on the target folder mapping
    for filename in os.listdir(source_folder):
        base_name = filename.split('_')[0]
        if base_name in target_name_map:
            new_name = target_name_map[base_name]
            old_file_path = os.path.join(source_folder, filename)
            new_file_path = os.path.join(source_folder, new_name)
            print(f"Renaming {old_file_path} to {new_file_path}")
            os.rename(old_file_path, new_file_path)
        else:
            # If no matching file, copy to the target folder
            source_file_path = os.path.join(source_folder, filename)
            target_file_path = os.path.join(target_folder, filename.replace("_", "-"))
            print(f"No matching file for {filename}. Copying from {source_file_path} to {target_file_path}")
            shutil.copy2(source_file_path, target_file_path)

def copy_missing_files(source_folder, destination_folder):
    count = 0
    # Ensure the destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # Get the set of files in the destination folder
    existing_files = set(os.listdir(destination_folder))

    # Iterate through files in the source folder
    for file in os.listdir(source_folder):
        if file not in existing_files:
            # File does not exist in destination folder, so copy it
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy2(source_path, destination_path)
            print(f"Copied {file} from {source_folder} to {destination_folder}")
        else:
            count += 1
    print(f"Number of existing files = {count}")

# Define your folders here
target_folder = 'datasets/style-coco/train2017/'
source_folder = 'datasets/ArtFlow-WCT/train2017/'

# Execute the function
rename_images(source_folder, target_folder)
copy_missing_files(target_folder, source_folder)
