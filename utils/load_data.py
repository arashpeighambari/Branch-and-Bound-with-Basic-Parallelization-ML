def load_data(filename):
    tsp_instances = []
    current_instance = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("TSP Instance"):
                if current_instance:
                    tsp_instances.append(current_instance)
                current_instance = []
            elif line:  # Skip empty lines
                try:
                    row = eval(line)  # Safely evaluate the string as a list
                    current_instance.append(row)
                except (SyntaxError, NameError):
                    print(f"Warning: Skipping invalid line: {line}")
        if current_instance:  # Add the last instance
            tsp_instances.append(current_instance)
    return tsp_instances
