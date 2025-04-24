
# Slot Attention Puzzle Solver

This project implements a Slot Attention-based model to solve symbolic puzzles where a 3×3 input grid is transformed into a 9×9 output grid. The transformation follows a specific logic that the model learns through supervised training.

---

## Project Workflow

### 1. Installing Dependencies

Install the required libraries using the following command:

```bash
pip install torch torchvision einops numpy pillow
```

**Dependencies Used:**
- `torch`, `torchvision`: Model definition, training, and data handling.
- `einops`: Efficient tensor reshaping and attention utilities.
- `numpy`: Data operations and transformations.
- `pillow`: Optional, used for visualizing matrices as images.

---

### 2. Generating Dataset

Start with a small number of raw examples stored in JSON format, each representing a pair of 3×3 input and 9×9 output matrices.

Then, use a `generate.py` script to generate 1000 additional examples based on the same transformation rule, ensuring a consistent pattern across samples.

---

### 3. Preparing the Dataset

Run `dataset.py` to convert the raw and generated samples into a PyTorch `Dataset` object. Each 3×3 input is expanded into a 48×48 image (each cell is a 16×16 square), while the 9×9 output remains a matrix target.

---

### 4. Defining the Model

Run `model.py` to build the Slot Attention model consisting of:

- A CNN encoder to extract features from input.
- A Slot Attention module that parses input into discrete symbolic slots.
- A decoder that reconstructs the full 9×9 matrix from the slots.

---

### 5. Training the Model

Run `train.py` to begin training. The training loop includes:

- Feeding the model input tensors.
- Comparing predicted vs. actual 9×9 outputs.
- Optimizing using MSE loss and Adam optimizer.

Since the dataset is pattern-specific and relatively small (2–5 base samples, extended to ~1000), training epochs can be adjusted as needed.

---

### 6. Inference and Testing

After training:

- Test the model using raw examples to confirm learning.
- Provide your own 3×3 inputs for prediction.

Example rule used during data generation:

```python
def tile_rule(input_grid):
    out = [[0]*9 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            if input_grid[i][j] != 0:
                for di in range(3):
                    for dj in range(3):
                        out[3*i+di][3*j+dj] = input_grid[di][dj]
    return out
```

You can run `inference.py` or a dedicated Colab notebook to visualize predictions.

---

## Logic Behind the Puzzle

The main transformation logic is as follows:

- If any cell in the 3×3 input is non-zero, copy the entire input into the corresponding 3×3 region in the 9×9 output.

This forms the basis for all generated training examples.

---

## Custom Input Inference

```python
my_input = [
    [2, 2, 2],
    [0, 0, 0],
    [0, 2, 2]
]

predicted = tile_rule(my_input)

for row in predicted:
    print(row)
```

Use this rule to verify if the Slot Attention model learns to generalize the puzzle logic effectively.
