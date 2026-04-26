# Analisis Sensitivitas dan Local Minima (Run Terbaru)

Sumber data analisis:
- `models/model_001/metrics.json`
- `models/model_001/soal_1/*/optuna_best.json`
- `models/model_001/soal_2/*/optuna_best.json`
- `models/model_001/soal_1/*/initial_selection.json`

## Konfigurasi Relevan
- Soal 1:
  - Optuna trial: `pso=36`, `ga=80`, `gradient_autograd=80`.
  - Parameter aktual sama untuk semua metode: `a_true=1.85`, `b_true=1.73`.
- Soal 2:
  - Optuna trial: `12` per metode.
  - Parameter aktual sama untuk semua metode: `a_true=0.22`, `b_true=4.40`.

## Hasil Ringkas

### Soal 1 (lebih sulit, banyak local minima)
- PSO (selected run):
  - `best_loss=0.003032`
  - `best_a=1.849846`, `best_b=1.729875` (sangat dekat ke aktual)
- GA (selected run):
  - `best_loss=0.184134`
  - `best_a=0.002586`, `best_b=0.113220` (jauh dari aktual)
- Gradient Autograd (selected run):
  - `best_loss=0.375580`
  - `best_a=1.95`, `best_b=1.95` (tidak menemukan basin global)

Catatan tuning value terbaik (bukan selected-run visual):
- PSO best trial value: `0.129909`
- GA best trial value: `0.129910` (mirip dengan PSO)
- Gradient best trial value: `0.252652` (lebih buruk)

### Soal 2
- PSO (selected run): `best_loss=0.004334`
- GA (selected run): `best_loss=0.004334`
- Gradient (selected run): `best_loss=0.004334`

Untuk soal 2, ketiga metode mampu mencapai loss yang hampir identik pada run terpilih.

## Jawaban Pertanyaan

### 1) Apakah GA lebih sensitif terhadap parameter dibanding PSO?
Kesimpulan dari run ini: **ya, cenderung lebih sensitif pada setup ini**, terutama di soal 1.

Indikasi:
- Walau trial GA untuk soal 1 sudah dinaikkan ke `80`, selected run masih tertahan di `best_loss=0.184134`.
- PSO dengan trial lebih sedikit (`36`) justru mampu mencapai `best_loss=0.003032` pada selected run.
- Pada domain sulit (local minima banyak), GA di run ini lebih mudah berakhir di solusi suboptimal.

Namun, ini tetap **empirical conclusion untuk konfigurasi saat ini**, bukan hukum universal untuk semua problem.

### 2) Apakah metaheuristik mengalahkan gradient-based untuk global minimum dan keluar dari jebakan local minima?
Kesimpulan dari run ini: **untuk soal 1, iya (terutama PSO), tapi tidak mutlak untuk semua soal**.

- Soal 1:
  - Metaheuristik (PSO) jelas unggul terhadap gradient-based pada run terpilih.
  - Gradient-based gagal keluar dari basin non-global yang terkait initial sulit.
- Soal 2:
  - Ketiga metode sama-sama mencapai solusi baik.
  - Jadi, tidak bisa diklaim metaheuristik selalu menang di semua landscape.

Kesimpulan praktis:
- Untuk objective non-konveks dengan banyak local minima, metaheuristik (khususnya PSO dalam eksperimen ini) lebih robust terhadap jebakan local minima.
- Gradient-based sangat bergantung pada basin awal dan bentuk landscape, sehingga lebih rentan stagnasi pada problem sulit tertentu.

## Implikasi untuk Eksperimen Lanjutan
- Jika target utama adalah global search pada landscape keras, prioritaskan PSO/GA dengan budget tuning yang memadai.
- Jika tetap ingin memakai gradient-based:
  - gunakan multi-start yang lebih luas,
  - tambah mekanisme restart/scheduler,
  - atau hybrid (metaheuristic warm-start lalu fine-tune gradient).
