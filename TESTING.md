# Testing

Jalankan test unit dari root folder `METAHEURISTIK`:

```powershell
./scripts/run_all_tests.ps1
```

Cakupan test saat ini:
- Validasi generator data sintetis.
- Validasi metrik MSE.
- Sanity check konvergensi optimizer PSO, GA, dan gradient_autograd pada data tanpa noise.
