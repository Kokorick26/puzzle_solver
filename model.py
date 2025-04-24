class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=64):
        super().__init__()
        self.num_slots, self.iters = num_slots, iters
        self.scale = dim**-0.5
        self.slots_mu    = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, num_slots, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        mu  = self.slots_mu.expand(B, -1, -1)
        sig = F.softplus(self.slots_sigma).expand(B, -1, -1)
        slots = mu + sig * torch.randn_like(mu)
        x = self.norm_input(x)
        k,v = self.to_k(x), self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)
            attn_logits = torch.einsum('bnd,bsd->bns', k, q)*self.scale
            attn = attn_logits.softmax(dim=1)
            updates = torch.einsum('bns,bnd->bsd', attn, v)
            slots = self.gru(updates.reshape(-1,D), slots_prev.reshape(-1,D)).reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

class SlotAutoEncoder(nn.Module):
    def __init__(self, res=(48,48), hidden=64, slots=9):
        super().__init__()
        C=3; H,W=res
        self.encoder = nn.Sequential(
            nn.Conv2d(C, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden, hidden,5,padding=2), nn.ReLU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, H*W, hidden))
        self.slot_attn = SlotAttention(slots, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden,5,padding=2), nn.ReLU(),
            nn.ConvTranspose2d(hidden, C, 5, padding=2), nn.Sigmoid()
        )

    def forward(self,x):
        B,C,H,W = x.shape
        f = self.encoder(x)                 
        tokens = (f.flatten(2).permute(0,2,1) + self.pos_emb)  
        slots = self.slot_attn(tokens)       
        out = 0
        for s in slots.permute(1,0,2):      
            feat = s.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
            out = out + self.decoder(feat)
        return out / slots.shape[1], slots  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SlotAutoEncoder(res=(48,48), hidden=64, slots=9).to(device)
opt = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()
