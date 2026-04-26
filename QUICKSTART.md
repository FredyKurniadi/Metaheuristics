# Quick Start

1. Setup environment:

```powershell
./scripts/setup_all.ps1
```

2. Jalankan eksperimen lengkap (2 soal x 3 metode: PSO, GA, gradient_autograd) dengan Optuna tuning aktif dari config:

```powershell
./scripts/run_experiment.ps1
```

Nilai `a_true`, `b_true`, `tuning_initial_params`, `evaluation_initial_params`, jumlah trial Optuna, dan search space ada di `train/configs/experiment.yaml`.
Visualisasi akan disimpan dari run konvergen paling lambat (atau fallback ke loss terbaik jika tidak ada run konvergen).

3. Lihat ringkasan metrik run terbaru:

```powershell
./scripts/show_latest_metrics.ps1
```

4. Jalankan test:

```powershell
./scripts/run_all_tests.ps1
```
