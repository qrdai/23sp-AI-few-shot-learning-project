#!/bin/bash

# Navigate to the test directory
cd ./dataset/Dataset_BUSI_with_GT/test

# Loop through all .png files in the directory
for file in *.png; do
  # Extract the number from the filename using grep
  number=$(echo "$file" | grep -o '[0-9]\+')
  
  # Rename the file with only the number followed by .png
  mv "$file" "${number}.png"
done

echo "All files have been renamed to contain only the number."
