# Installation

## Prasyarat
- Windows + PowerShell.
- Python 3.10 atau lebih baru (disarankan 3.11).

## Setup Environment
Dari root folder `METAHEURISTIK`:

```powershell
./scripts/setup_all.ps1
```

Script setup akan:
- Membuat `.venv` di dalam folder `METAHEURISTIK`.
- Upgrade `pip`.
- Install dependencies dari `train/requirements.txt`.
- Menyimpan versi Python, pip, dan library ke `docs/REPRODUCIBILITY.md`.
