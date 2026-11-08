# Injection moulding capability twin

Research code and a saved analysis notebook accompanying:

**On-demand capability forecasting in injection moulding: a state-conditioned digital twin with audited cycle-level predictive bands — a case study on cushion monitoring**

Authors: Mohammed Hdid, Hakim Jebari and Hassan Samadi.

The notebook investigates cushion forecasting and conditional capability assessment using recurrent and temporal convolutional models. The standalone generator provides an illustrative synthetic dataset with the same input column names.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Notebooks/cushion_capability_forecasting.ipynb` | Analysis code with saved research outputs and figures |
| `Scripts/synthetic_cushion_generator.py` | Standalone synthetic cushion data generator |
| `requirements.txt` | Python dependencies for the notebook |
| `CITATION.cff` | Citation metadata for this software |
| `LICENSE` | MIT licence, with an explicit CC0 exception for the generator |

## View the saved results

Open the notebook on GitHub, or download it and open it in Google Colab or Jupyter. **Viewing the saved outputs does not require running any cells.** Keep an unchanged copy of the notebook if you want to retain those outputs.

The industrial input dataset is not included. Consequently, the public repository alone cannot reproduce the numerical results reported in the manuscript. Synthetic data are for demonstration and experimentation; they were not used to produce the reported industrial results.

## Generate demonstration data

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install numpy pandas
python Scripts/synthetic_cushion_generator.py --out synthetic_cushion.csv --verify
```

The default generator produces 25,767 synthetic rows. For a smaller example:

```bash
python Scripts/synthetic_cushion_generator.py --n-cycles 5000 --seed 42 --out synthetic_cushion.csv
```

The output is a semicolon-separated CSV with day-first timestamps. The generator uses illustrative stochastic assumptions and reference summary statistics; its verification output is a diagnostic comparison, not evidence that it reproduces the industrial dataset or manuscript findings.

## Experiment with the notebook

1. Make a separate working copy of the notebook before running it.
2. Generate the synthetic CSV and upload it to your Colab runtime.
3. In the working copy, set `DATA_PATH` to the uploaded file, for example `/content/synthetic_cushion.csv`, and retain `DELIMITER = ";"`.
4. Adapt the Google Drive mounting and path setup to your runtime, then execute the working copy in order.

Synthetic experiments will replace saved outputs in that working copy and will produce different numerical results. Some analyses require sufficiently long segments; a small demonstration dataset may not support every diagnostic.

For a local Python environment:

```bash
python -m pip install -r requirements.txt
```

Open the notebook with your preferred Jupyter-compatible application. The notebook contains Colab-specific setup, including `google.colab` and Google Drive mounting; adapt those setup cells for local execution. `google.colab` is supplied by Colab and is not a portable dependency in `requirements.txt`.

### Input schema

The notebook expects these columns:

```text
timestamp
Maximum_Injection_Pressure
Switch_Over_Pressure
Injection_Time
Dosing_Time
Back_pressure_ActValue
SwitchOver_volume_ActValue
Dosing_Volume_ActValue
Material_Cushion
```

Do not commit confidential industrial CSV files or notebook outputs that display individual industrial records. Running the data-preview cells on private data can display records again.

### Recorded research environment

The saved notebook records Python 3.12.13, PyTorch 2.11.0+cu128, NumPy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, SciPy 1.16.3, statsmodels 0.14.6 and SHAP 0.52.0, with a Tesla T4 GPU. These are historical environment values recorded in the outputs. The requirements file pins those Python packages where recorded, while leaving plotting and widget packages unpinned because their study versions were not recorded. It is not a complete environment lockfile. CUDA installation depends on the host system.

## Interpretation

The study concerns cushion monitoring in a single industrial case. Cushion capability is not direct validation of finished-part quality. Conditional forecasts depend on their stated recipe and operating assumptions; what-if scenarios do not establish causal effects. Coverage audits concern cycle-level predictive bands and do not by themselves establish calibrated confidence intervals for capability indices.

## Citation

The initial research release, **v1.0.0**, was archived on Zenodo on **8 November 2026**:

**DOI: [10.5281/zenodo.24060142](https://doi.org/10.5281/zenodo.24060142)**

Hdid, M., Jebari, H., & Samadi, H. (2026). *Injection moulding capability twin: cushion capability forecasting* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24060142

Use GitHub's “Cite this repository” option, based on `CITATION.cff`. This DOI identifies the archived software release, not a journal article. Cite the article separately when its publication details are available. The `main` branch can contain later documentation updates; use the archived release when referring to v1.0.0.

## Licence

The repository code and documentation are offered under the MIT licence, except `Scripts/synthetic_cushion_generator.py`, which retains its existing **CC0-1.0** dedication. See `LICENSE` and the generator header. No rights to the industrial dataset are granted by these software licences.
