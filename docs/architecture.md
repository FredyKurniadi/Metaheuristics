# Arsitektur

## Alur Utama
1. Baca konfigurasi YAML.
2. Generate data sintetis untuk tiap soal.
3. Jalankan optimizer (PSO / GA) untuk estimasi `(a, b)`.
4. Simpan history iterasi dan metrik akhir.
5. Buat visualisasi konvergensi statis + animasi.
6. Simpan artefak ke `models/model_xxx/`.

## Komponen Kode
- `train/src/core.py`: fungsi target, generator data, noise, normalisasi, metrik MSE.
- `train/src/optimizers.py`: implementasi PSO, GA, dan gradient-based autograd.
- `train/src/visualize.py`: plot loss landscape, animasi path `(a,b)`, animasi kurva `y_pred`.
- `train/src/experiment.py`: orkestrasi satu eksperimen.
- `train/src/main.py`: entrypoint untuk menjalankan semua kombinasi.

## Reproducibility
- Semua random generator menggunakan seed dari konfigurasi.
- Snapshot konfigurasi disimpan pada setiap run model.
- Environment versions dicatat pada `docs/REPRODUCIBILITY.md`.
