#!/bin/bash

# Navigate to the specific directory
cd ./dataset/Dataset_BUSI_with_GT/val/normal

# Use find to locate and delete files that contain '<the given keyword>' in their name
# find . -type f -name '*normal*.png'
find . -type f -name '*normal*.png' -exec rm {} +

echo "All images with the given keyword in the filename have been deleted."
