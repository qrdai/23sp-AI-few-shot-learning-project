#!/bin/bash

# Define the parent directory where the files will be collated
parent_directory="unlabel"

# Create the parent directory if it doesn't exist
mkdir -p "$parent_directory"

# Variable to track the incremental number
counter=1

# Array of directory paths
directory_paths=(
  "/remote-home/share/course23/aicourse_dataset_final/10shot_cifar100_20200721/unlabel"
  "/remote-home/share/course23/aicourse_dataset_final/10shot_country211_20210924/unlabel"
  "/remote-home/share/course23/aicourse_dataset_final/10shot_food_101_20211007/unlabel"
  "/remote-home/share/course23/aicourse_dataset_final/10shot_oxford_iiit_pets_20211007/unlabel"
  "/remote-home/share/course23/aicourse_dataset_final/10shot_stanford_cars_20211007/unlabel"
)

# Iterate over the directory paths
for directory_path in "${directory_paths[@]}"; do
  # Check if the directory exists
  if [ -d "$directory_path" ]; then
    # Iterate over all files in the directory
    find "$directory_path" -type f -print0 | while IFS= read -r -d '' file; do
      # Extract the file extension
      extension="${file##*.}"

      # Construct the new file name with the incremental number
      new_file_name="$parent_directory/$counter.$extension"

      # Increment the counter
      counter=$((counter + 1))

      # Copy the file to the parent directory with the new name
      cp "$file" "$new_file_name"
      echo "Copied $file to $new_file_name"
    done
  else
    echo "Directory '$directory_path' does not exist."
  fi
done
