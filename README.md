# llm-clinical-preds-gender
Experiment: How do LLM predictions when perturbing gender in clinical notes?

* `llm_gender_swap.py`: Formats MIMIC-IV notes for prompting LLM, generates a spreadsheet with model outputs
* `gender_swap_scoring.py`: Computes precision, recall, and accuracy of LLM outputs compared to human annotations
* `parse_mimic.py`: Parses the MIMIC-IV notes file, generates a spreadsheet with desired note info
* `run_experiments_finalish.ipynb`: Generates the predictions from various original and transformed MIMIC-IV notes
* `run_analysis.ipynb`: Generates the statistical results and figures
