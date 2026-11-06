# Karakoram Glacier Anomaly

**Investigating glacier mass balance, climate variability, and spatial patterns of glacier change across High Mountain Asia.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Remote Sensing](https://img.shields.io/badge/Remote%20Sensing-Satellite%20Data-2D6A8A?style=flat)](#data-sources)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

The Karakoram Anomaly describes the unusual behaviour of glaciers in the Karakoram, where many glaciers historically maintained relatively stable or slightly positive mass balances despite widespread glacier retreat elsewhere.

This project investigates glacier change across the Karakoram and neighbouring mountain regions, examining how glacier mass balance varies geographically and over time, and how these patterns relate to climate variability.

Using satellite observations, glacier inventories, historical mass-balance measurements, and climate reanalysis data, the project brings together remote sensing, geospatial analysis, statistical modelling, and machine learning.

The broader objective is to understand whether the Karakoram Anomaly represents a persistent regional phenomenon or a more spatially variable response to changing climatic conditions.

<img width="1920" height="995" alt="high_mountain_asia_glaciers" src="https://github.com/user-attachments/assets/cd869b09-0c2f-4d39-9113-f7454d767170" />

<img width="1489" height="790" alt="glacier_cdf" src="https://github.com/user-attachments/assets/a85882e7-b581-4408-b108-4bced89de5b0" />


## Research questions

- How has glacier mass balance evolved across the Karakoram over recent decades?
- Does the Karakoram exhibit different behaviour from neighbouring mountain regions?
- Which regions show evidence of glacier stability, accumulation, or mass loss?
- How are temperature and precipitation associated with observed mass-balance patterns?
- Can historical observations help identify potential future trends in glacier change?

## Project components

### 1. Glacier inventory and satellite observations

Exploration of glacier outlines and satellite-derived observations using the Randolph Glacier Inventory (RGI 7.0) and Sentinel-2 imagery.

### 2. Historical mass-balance analysis

Processing glacier mass-balance observations from the World Glacier Monitoring Service (WGMS).

The analysis examines annual and cumulative specific mass-balance records across different mountain regions, allowing comparisons of long-term glacier behaviour.

### 3. Spatial analysis across High Mountain Asia

Mountain regions are divided into spatial analysis grids to investigate the distribution of glacier mass balance.

Regional comparisons include the Karakoram, Western Kunlun, Pamir, Himalaya, and other High Mountain Asia regions.

### 4. Climate data integration

ERA5 and ERA5-Land climate reanalysis data are incorporated into spatial analysis grids.

Temperature and precipitation variables are used to investigate possible relationships between climatic variability and glacier mass balance.

### 5. Machine learning and time-series modelling

Exploratory modelling using statistical and machine-learning approaches to investigate historical trends and forecast potential changes.

Models explored include:
- Random Forest Regression
- ARIMA
- XGBoost

Modelled results are compared against observed historical records. Forecasts are exploratory and should not be interpreted as physically based glacier projections.

### 6. Interactive geospatial visualisation

Interactive maps provide a spatial overview of glacier mass-balance patterns across mountain regions.

The maps allow users to explore individual spatial grids, compare regional behaviour, and examine the geographical distribution of positive and negative mass-balance observations.

## Data sources

| Dataset | Purpose |
|---|---|
| [Randolph Glacier Inventory (RGI 7.0)](https://www.glims.org/RGI/) | Glacier outlines and inventory |
| [Sentinel-2](https://dataspace.copernicus.eu/) | Satellite imagery and surface observations |
| [WGMS](https://wgms.ch/) | Historical glacier mass-balance observations |
| [ERA5](https://cds.climate.copernicus.eu/) | Historical climate reanalysis |
| [ERA5-Land](https://cds.climate.copernicus.eu/) | Land-surface climate variables |

## Technology

- **Python:** Data processing and analysis
- **Pandas / NumPy:** Data manipulation
- **GeoPandas / Shapely:** Geospatial processing
- **Matplotlib:** Data visualisation
- **Folium:** Interactive mapping
- **Scikit-learn:** Machine learning
- **Statsmodels:** Time-series analysis
- **XGBoost:** Gradient boosting
- **Sentinel Hub:** Satellite imagery access
- **Copernicus Climate Data Store:** Climate data retrieval

## Repository structure

```text
karakoram-glacier-anomaly/
│
├── maps/
│
├── sentinel_rgi7_exploration.ipynb
├── wgms_exploration.ipynb
├── wgms_exploration.py
│
├── LICENSE
└── README.md
```

*The repository structure will evolve as the analysis develops.*

## Getting started

Clone the repository:

```bash
git clone https://github.com/zaidi2003/karakoram-glacier-anomaly.git

cd karakoram-glacier-anomaly
```

Create a virtual environment:

```bash
python -m venv .venv

source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install numpy pandas matplotlib geopandas shapely folium scikit-learn statsmodels xgboost jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Some workflows require additional dependencies and access credentials for Copernicus data services.

## Project status

This is an ongoing research and exploration project. Data processing workflows, regional comparisons, climate integration, and predictive experiments are being developed iteratively.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
