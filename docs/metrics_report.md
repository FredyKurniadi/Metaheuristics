# Metrics Report

File ini menjadi ringkasan metrik run terakhir.

Setelah menjalankan:

```powershell
./scripts/run_experiment.ps1
```

cek metrik terbaru dengan:

```powershell
./scripts/show_latest_metrics.ps1
```

Metrik utama:
- `best_loss` (MSE)
- `best_a`
- `best_b`
- `iterations`
- `population`

Analisis interpretasi hasil local minima dan sensitivitas metode tersedia di:
- `docs/analysis_local_minima.md`
