with open("pattern.json") as f:
    raw = json.load(f)["train"]

def grid_to_img(grid, cell=16):
    H,W = 3,3
    img = Image.new("RGB", (W*cell, H*cell), (0,0,0))
    draw = ImageDraw.Draw(img)
    cmap = [(i*36,)*3 for i in range(8)]
    for i in range(H):
        for j in range(W):
            val = grid[i][j]
            color = cmap[val] if val< len(cmap) else (255,255,255)
            draw.rectangle([j*cell,i*cell,(j+1)*cell,(i+1)*cell], fill=color)
    return img

class InputOnly(Dataset):
    def __init__(self, samples, transform):
        self.imgs = [transform(grid_to_img(s["input"])) for s in samples]
    def __len__(self):    return len(self.imgs)
    def __getitem__(self,i):
        x = self.imgs[i]
        return x, x  

transform = transforms.Compose([
    transforms.ToTensor(),  
])

ds = InputOnly(raw, transform)
loader = DataLoader(ds, batch_size=4, shuffle=True)
