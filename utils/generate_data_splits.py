import random
from pathlib import Path

# Specify the paths to GT and InputData directories
# gt_path = Path('/media/davidg-dl/Second SSD/DATASET_test/GT/SDFs')
gt_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/GT/SDFs')
input_data_path = Path('/media/davidg-dl/Second SSD/CompleteDataset/InputData/SDFs')
# input_data_path = Path("/media/davidg-dl/Second SSD/DATASET_test/InputData/SDFs")

# Define the split percentages
train_split = 0.7  # 70% for training
val_split = 0.20  # 20% for validation
test_split = 0.10  # 10% for testing

# Define the number of samples for the overfit split
overfit_samples = 10 * 7

# Open the original output file
original_output_path = Path("output-final.txt")
with open(original_output_path, "r") as original_output_file:
    # Read the lines from the original output file
    lines = original_output_file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the number of samples for each split
    total_samples = len(lines)
    train_samples = int(total_samples * train_split)
    val_samples = int(total_samples * val_split)
    test_samples = total_samples - train_samples - val_samples

    # Split the lines into train, val, and test lists
    train_lines = lines[:train_samples]
    val_lines = lines[train_samples: train_samples + val_samples]
    test_lines = lines[train_samples + val_samples:]

    # Shuffle the lines within each split
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    random.shuffle(test_lines)

    # Create the train split file
    train_output_path = Path("train.txt")
    with open(train_output_path, "w") as train_output_file:
        train_output_file.writelines(train_lines)

    # Create the val split file
    val_output_path = Path("val.txt")
    with open(val_output_path, "w") as val_output_file:
        val_output_file.writelines(val_lines)

    # Create the test split file
    test_output_path = Path("test.txt")
    with open(test_output_path, "w") as test_output_file:
        test_output_file.writelines(test_lines)

    # Create the overfit split file
    overfit_output_path = Path("overfit.txt")
    with open(overfit_output_path, "w") as overfit_output_file:
        # Choose the specified number of samples for the overfit split
        overfit_lines = random.sample(lines, overfit_samples)
        overfit_output_file.writelines(overfit_lines)

print("Splitting completed.")
