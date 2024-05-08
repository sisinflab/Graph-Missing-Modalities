import re

# Open the file for reading
file_path = "your_file_path.txt"  # Replace with your file path
best_model_results = []

# Regular expression pattern to extract the best model results
pattern = r"I Best Model results: {(\d+): {'Recall': ([\d.]+), 'nDCG': ([\d.]+), 'Precision': ([\d.]+)}, (\d+): {'Recall': ([\d.]+), 'nDCG': ([\d.]+), 'Precision': ([\d.]+)}, (\d+): {'Recall': ([\d.]+), 'nDCG': ([\d.]+), 'Precision': ([\d.]+)}}"

with open(file_path, 'r') as file:
    for line in file:
        match = re.match(pattern, line)
        if match:
            results = {
                int(match.group(1)): {
                    'Recall': float(match.group(2)),
                    'nDCG': float(match.group(3)),
                    'Precision': float(match.group(4))
                },
                int(match.group(5)): {
                    'Recall': float(match.group(6)),
                    'nDCG': float(match.group(7)),
                    'Precision': float(match.group(8))
                },
                int(match.group(9)): {
                    'Recall': float(match.group(10)),
                    'nDCG': float(match.group(11)),
                    'Precision': float(match.group(12))
                }
            }
            best_model_results.append(results)

print(best_model_results[0])
