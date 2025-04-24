img, _ = ds[0]
x = img.unsqueeze(0).to(device)  # [1,3,48,48]
model.eval()
with torch.no_grad():
    recon, masks = model(x)


grid3 = raw[0]["input"]


def tile_rule(input_grid):
    out = [[0]*9 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            if input_grid[i][j]!=0:
                for di in range(3):
                    for dj in range(3):
                        out[3*i+di][3*j+dj] = input_grid[di][dj]
    return out

pred = tile_rule(grid3)
print("Predicted:\n", pred)
print("Ground-truth:\n", raw[0]["output"])




def tile_rule(input_grid):
    """
    For each non‑zero in the 3×3 input, copies the entire 3×3 block
    into the corresponding 3×3 region of the 9×9 output.
    """
    out = [[0]*9 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            if input_grid[i][j] != 0:
                for di in range(3):
                    for dj in range(3):
                        out[3*i+di][3*j+dj] = input_grid[di][dj]
    return out

my_input = [
    [2, 2, 2],
    [0, 0, 0],
    [0, 2, 2]
]

predicted = tile_rule(my_input)

print("🧩 Input (3×3):")
for row in my_input:
    print(row)

print("\n↳ Predicted 9×9 output:")
for row in predicted:
    print(row)
