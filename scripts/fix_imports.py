import os
import re

domains = [
    "business", "descriptive", "geospatial", "graphs",
    "inferential", "ml_support", "nlp", "relational",
    "segmentation", "survival", "time_series"
]

statistics_dir = r"c:\Users\justi\Desktop\EDA-project\statistics"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    for domain in domains:
        # replace `from domain.` with `from statistics.domain.`
        content = re.sub(fr'^from {domain}\.', f'from statistics.{domain}.', content, flags=re.MULTILINE)
        content = re.sub(fr'^import {domain}\.', f'import statistics.{domain}.', content, flags=re.MULTILINE)
        content = re.sub(fr' from {domain}\.', f' from statistics.{domain}.', content, flags=re.MULTILINE)

    if content != original_content:
        print(f"Fixing {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

for root, _, files in os.walk(statistics_dir):
    for file in files:
        if file.endswith('.py'):
            process_file(os.path.join(root, file))

print("Imports fixed!")
