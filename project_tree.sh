#!/bin/bash

# Script: project_tree.sh
# Purpose: Print the project directory tree excluding ./venv

# If tree command exists, use it
if command -v tree &> /dev/null; then
    tree -I 'venv'
else
    # Fallback to find + sed formatting
    echo "tree command not found, using fallback method..."
    find . -path "./venv" -prune -o -print | sed -e 's/^\.\///' | awk -F/ '
    {
        indent = length($0) - length(gensub(/[^\/]/, "", "g"))
        printf("%*s%s\n", indent*4, "", $NF)
    }'
fi
