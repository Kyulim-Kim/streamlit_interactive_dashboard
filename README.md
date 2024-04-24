# Interactive Dashboard using Streamlit
## 1. Download Dataset
dataset can also be downloaded at [NHIS-2018](https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm)
```bash
bash dataset.sh
```

## 2. Run Streamlit App
make sure proper anaconda python interpreter is on set.
```bash
conda env create -f environment.yml
source activate nhis_dashboard
streamlit run app.py
```
