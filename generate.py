def transform_3x3_to_9x9(input_grid):
    out = [[0]*9 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            if input_grid[i][j] != 0:
                for di in range(3):
                    for dj in range(3):
                        out[3*i+di][3*j+dj] = input_grid[di][dj]
    return out

def gen_input(nonzero_vals=[2,4,6,7]):
    grid = [[0]*3 for _ in range(3)]
    coords = [(i,j) for i in range(3) for j in range(3)]
    k = random.randint(1,9)
    for (i,j) in random.sample(coords, k):
        grid[i][j] = random.choice(nonzero_vals)
    return grid

def create_dataset(train_n=1000, test_n=200, out="pattern.json"):
    train, test = [], []
    for _ in range(train_n):
        inp = gen_input()
        train.append({"input": inp, "output": transform_3x3_to_9x9(inp)})
    for _ in range(test_n):
        inp = gen_input()
        test.append({"input": inp, "output": transform_3x3_to_9x9(inp)})
    with open(out,"w") as f:
        json.dump({"train": train, "test": test}, f, indent=2)
    print(f"→ saved {train_n} train + {test_n} test to {out}")

create_dataset(train_n=1000, test_n=200)
