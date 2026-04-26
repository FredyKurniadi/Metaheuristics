# METAHEURISTIK - Estimasi Parameter dengan PSO, GA, dan Gradient-Based

Proyek ini membangun eksperimen optimisasi metaheuristik untuk mencari parameter `a` dan `b` pada dua fungsi nonlinier berbasis data sintetis dengan noise.

## Tujuan
- Mengestimasi `(a, b)` pada 2 soal fungsi sintetis.
- Membandingkan konvergensi PSO vs GA vs gradient-based (autograd).
- Menyediakan visualisasi konvergensi yang mudah dianalisis.

## Fungsi yang Diselesaikan
1. `y = sin(a x) cos(b x) + noise`
2. `y = exp(-a x^2) sin(b x) + noise`

Rentang parameter:
- Soal 1: `a` di `[0, 2]`, `b` di `[0, 2]`
- Soal 2: `a` di `[-5, 5]`, `b` di `[-5, 5]`

Catatan setup eksperimen:
- Soal 1 menggunakan rentang `x` yang otomatis mencakup 1 periode utuh `sin(a x) cos(b x)`.
- Tiap soal punya satu `a_true,b_true` yang sama untuk semua metode.
- Optuna tuning per soal-metode dilakukan pada initial yang sama (`tuning_initial_params`).
- Setelah tuning, tiap metode diuji pada beberapa initial berbeda (`evaluation_initial_params`) yang jauh dari nilai aktual.
- Hyperparameter PSO, GA, dan gradient_autograd dituning dengan Optuna.
- Artefak visualisasi yang disimpan adalah run konvergen paling lambat (dengan hyperparameter terbaik); fallback ke loss terbaik jika tidak ada yang konvergen.

## Struktur
- `datasets/`: data sintetis `raw` dan `processed`.
- `docs/`: dokumen arsitektur, dataset, training, metrik, dan reproducibility.
- `models/`: output hasil run (`model_001`, `model_002`, dst).
- `scripts/`: setup environment, run eksperimen, test, dan ringkasan metrik.
- `train/configs/`: konfigurasi eksperimen.
- `train/src/`: implementasi Python.
- `train/tests/`: test unit.

## Output per Kombinasi Soal x Metode
- Plot loss landscape + learning path `(a, b)`.
- Video/GIF pergerakan estimasi `(a, b)` per iterasi.
- Video/GIF perubahan `y_pred` per iterasi dibanding `y_true` (fungsi tetap).
- Ringkasan metrik akhir (`best_a`, `best_b`, `best_loss`).
- Ringkasan tuning Optuna (`optuna_best.json`).

## Command Utama
Jalankan dari folder `METAHEURISTIK`:

```powershell
./scripts/setup_all.ps1
./scripts/run_experiment.ps1
./scripts/show_latest_metrics.ps1
./scripts/run_all_tests.ps1
```

Lihat `QUICKSTART.md` untuk langkah cepat.
