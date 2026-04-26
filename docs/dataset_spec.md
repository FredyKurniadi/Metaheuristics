# Dataset Specification

## Sumber Data
Data dibuat sintetis pada saat runtime.

## Fitur
- Input: `x` (1D).
- Target: `y`.

## Soal 1
`y = sin(a x) cos(b x) + noise`

## Soal 2
`y = exp(-a x^2) sin(b x) + noise`

## Konfigurasi Data
- `x_range`: `[x_min, x_max]`
- `num_samples`: jumlah titik data
- `noise.type`: `gaussian` atau `uniform`
- `noise.params`: parameter sesuai tipe noise

## Penyimpanan
Snapshot data observasi per eksperimen disimpan ke:
- `models/model_xxx/<soal>/<metode>/data_points.csv`
