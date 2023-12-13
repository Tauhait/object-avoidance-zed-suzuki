import itertools
import csv

# Define the constant values and corresponding string values for each variable
variables = {
    "CollisionWarning": {
        100: "NO_WARN",
        101: "MID_WARN",
        102: "URGENT_WARN"
    },
    "CollisionAvoidance": {
        200: "OVERTAKE",
        201: "CONTINUE",
        202: "SWITCH"
    },
    "LaneStates": {
        300: "DRIVING",
        301: "CHANGE",
        302: "OVERTAKE"
    },
    "OpticalFlow": {
        400: "TRAFFIC_LEFT",
        401: "SAFE_TO_OVERTAKE",
        402: "TRAFFIC_RIGHT"
    }
}

# Create lists of the constant values for each variable
CollisionWarning = [100, 101, 102]
CollisionAvoidance = [200, 201, 202]
LaneStates = [300, 301, 302]
OpticalFlow = [400, 401, 402]

# Create a list of all variables
all_variables = [CollisionWarning, CollisionAvoidance, LaneStates, OpticalFlow]

# Generate all possible combinations
combinations = list(itertools.product(*all_variables))

# Specify the output CSV file path
csv_file_path = "combinations.csv"

# Write the combinations to the CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row with variable names
    header = ["CollisionWarning", "CollisionAvoidance", "LaneStates", "OpticalFlow"]
    csv_writer.writerow(header)
    
    # Write each combination as a row in the CSV file
    for combo in combinations:
        combo_str = [variables[var_name][value] for var_name, value in zip(variables.keys(), combo)]
        csv_writer.writerow(combo_str)

print(f"Combinations written to {csv_file_path}")
