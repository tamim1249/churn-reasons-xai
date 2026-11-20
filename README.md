# Data

All datasets are publicly available on Kaggle and are **not** redistributed in this repository.
Download them (or attach them in a Kaggle notebook with **+ Add Input**) before running the pipeline.

| Used in | Dataset | Kaggle | File(s) |
|---|---|---|---|
| Steps 01–07, 09, 10 (Study 1) | IBM Telco Customer Churn sample, 33 variables | [yeanzc/telco-customer-churn-ibm-dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) | `Telco_customer_churn.xlsx` |
| Steps 08, 10 (Study 2) | IBM Telco Customer Churn, release 11.1.3+, 50 variables | [alfathterry/telco-customer-churn-11-1-3](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3) | `telco.csv` |
| Step 09 | Orange / BigML telecom churn | [mnassrib/telecom-churn-datasets](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) | `churn-bigml-80.csv`, `churn-bigml-20.csv` |
| Step 09 | Cell2Cell (Duke/Teradata churn tournament) | [jpacse/datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom) | `cell2celltrain.csv` |
| Step 09 | Iranian telecom churn (UCI) | [royjafari/customer-churn](https://www.kaggle.com/datasets/royjafari/customer-churn) | `Customer Churn.csv` |

## Local layout

Place the files anywhere under `data/raw/` (sub-folders are fine), for example:

```
data/raw/
├── Telco_customer_churn.xlsx
├── telco.csv
├── churn-bigml-80.csv
├── churn-bigml-20.csv
├── cell2celltrain.csv
└── Customer Churn.csv
```

Files are located by their column names, not by their paths. Steps 08 and 09 download missing datasets automatically with `kagglehub` if Kaggle API credentials are configured.
