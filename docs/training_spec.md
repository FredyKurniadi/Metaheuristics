# Training Specification

## Ruang Parameter
- Soal 1:
  - `a` dalam `[0, 2]`
  - `b` dalam `[0, 2]`
- Soal 2:
  - `a` dalam `[-5, 5]`
  - `b` dalam `[-5, 5]`

## Inisialisasi Eksperimen
- `tuning_initial_params`: initial tunggal per soal (dipakai sama oleh semua metode saat tuning Optuna).
- `evaluation_initial_params`: daftar initial berbeda per soal untuk evaluasi pasca-tuning.
- Nilai awal dipilih cukup jauh dari parameter aktual agar pembelajaran optimizer terlihat jelas.

## Hyperparameter Tuning
- Tuning dilakukan dengan Optuna untuk tiap kombinasi `{soal, metode}`.
- Metrik objective tuning adalah MSE minimum.
- Trial tuning tidak menyimpan GIF/landscape.
- Hanya trial terbaik yang dijalankan ulang untuk menyimpan artefak visualisasi.
- Dengan hyperparameter terbaik, jalankan ulang pada beberapa initial.
- Pilih run yang konvergen paling lambat untuk divisualisasikan.
- Jika tidak ada run yang konvergen, gunakan run dengan loss akhir terbaik.

## Objective
- MSE antara `y_observed` dan `y_pred(a, b)`.

## Optimizer
- PSO:
  - `population`
  - `iterations`
  - `w`, `c1`, `c2`
- GA:
  - `population`
  - `iterations`
  - `elite_ratio`
  - `mutation_rate`
  - `mutation_scale`
- Gradient-based (autograd):
  - `iterations`
  - `learning_rate`
  - `beta1`
  - `beta2`
  - `epsilon`

## Artifact
Setiap kombinasi eksperimen menghasilkan:
- `history.csv`
- `summary.json`
- `loss_landscape_path.png`
- `ab_path.gif`
- `y_pred_vs_true.gif`
- `optuna_best.json`
- `initial_selection.json`
