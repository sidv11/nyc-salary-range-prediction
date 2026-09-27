# About the data in this project

The raw and processed data files aren't included in this repository.

This project used a job-postings dataset provided as part of a client research assignment, and the terms of that assignment don't allow redistributing the client's copy of the data on any public platform. That's a confidentiality rule about *their copy* of the file, not about the dataset itself — the same dataset is separately published as open data.

## Getting the same dataset yourself

The file used here is "Jobs NYC Postings," published by NYC Open Data:

- Dataset page: https://data.cityofnewyork.us/d/kpav-sd4t
- Direct CSV download: https://data.cityofnewyork.us/api/views/kpav-sd4t/rows.csv?accessType=DOWNLOAD

Download the CSV and save it as:

```
data/raw/Jobs_NYC_Postings.csv
```

Run the notebook (`notebook/Salary_Range_Prediction.ipynb`) from the top, and it will clean the data, engineer features, save the processed version to `data/processed/jobs_cleaned.csv`, and train the models exactly as described in the write-up. Row counts may differ slightly from the ones quoted in this project, since NYC Open Data updates the live file over time and this project used a snapshot from September 2026.

## What's excluded and why

| Folder | What's here | What's missing |
|---|---|---|
| `data/raw/` | `data_dictionary.txt` (column descriptions — not the data itself) | The actual `Jobs_NYC_Postings.csv` / `.xlsx` files |
| `data/processed/` | (empty until you run the notebook) | `jobs_cleaned.csv`, since it's derived directly from the raw file |
| `predictions/` | `model_scores.csv` (aggregate R²/MAE/RMSE only) | `salary_predictions.csv`, since it lists real Job IDs, titles, agencies, and actual salaries row by row |

The trained models in `models/` and the full write-up in `report/` don't contain any raw data rows, so they're included as-is.
