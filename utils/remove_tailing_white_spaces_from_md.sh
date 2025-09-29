#!/bin/bash

# Check if file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <file-path>"
  exit 1
fi

file="$1"

# Use sed to remove trailing whitespace in-place
sed -i 's/[[:space:]]\+$//' "$file"

echo "Trailing whitespace removed from $file"