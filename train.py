EPOCHS=50
for ep in range(1, EPOCHS+1):
    model.train(); L=0
    for x, _ in loader:
        x = x.to(device)
        recon, slots = model(x)
        loss = criterion(recon, x)
        opt.zero_grad(); loss.backward(); opt.step()
        L += loss.item()
    print(f"Ep{ep:02d} ↓ Loss {L/len(loader):.4f}")
