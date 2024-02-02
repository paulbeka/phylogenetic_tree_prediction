#!/bin/sh

# Run your Python script
python3 /app/src/main.py -m data_generation -l src/data/training -o /app/src/output

chmod 644 /app/src/output/data_generation_output.pickle

# Optional- keep docker running
tail -f /dev/null