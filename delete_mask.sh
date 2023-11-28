#!/bin/bash

# Navigate to the dataset directory
cd ./dataset/Dataset_BUSI_with_GT

# Use find to search for and delete files that contain 'mask' in their name
# find . -type f -name '*mask*.png'
find . -type f -name '*mask*.png' -exec rm {} \;

echo "All mask images have been deleted."
