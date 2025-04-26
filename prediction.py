import numpy as np
import matplotlib.pyplot as plt

arc_colors = [
    (0, 0, 0),         # 0: Black
    (255, 0, 0),       # 1: Red
    (0, 255, 0),       # 2: Green
    (255, 255, 0),     # 3: Yellow
    (0, 0, 255),       # 4: Blue
    (255, 0, 255),     # 5: Magenta
    (0, 255, 255),     # 6: Cyan
    (255, 165, 0),     # 7: Orange
    (128, 0, 128),     # 8: Purple
    (128, 128, 128),   # 9: Gray
]

def show_colored_grid(grid, title="Grid"):
    grid = np.array(grid)
    h, w = grid.shape
    rgb_grid = np.zeros((h, w, 3), dtype=np.uint8)
    for val in range(10):
        rgb_grid[grid == val] = arc_colors[val]
    plt.imshow(rgb_grid)
    plt.title(title)
    plt.axis('off')
    plt.show()

def tile_rule(input_grid):
    grid_height = len(input_grid)
    grid_width = len(input_grid[0])

    out_height = grid_height * 3
    out_width = grid_width * 3
    out = [[0] * out_width for _ in range(out_height)]

    for i in range(grid_height):
        for j in range(grid_width):
            if input_grid[i][j] != 0:
                for di in range(grid_height):
                    for dj in range(grid_width):
                        out[grid_height * i + di][grid_width * j + dj] = input_grid[di][dj]
    return out

if __name__ == "__main__":
    my_input = [
       [6,4,4],
       [2,4,6],
       [4,2,7]
    ]

    predicted_output = tile_rule(my_input)

    print("Input Grid:")
    for row in my_input:
        print(row)

    print("\n Predicted Output Grid:")
    for row in predicted_output:
        print(row)

    show_colored_grid(my_input, title="Input Grid")
    show_colored_grid(predicted_output, title="Predicted Output Grid")
