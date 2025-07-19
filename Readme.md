# Machine Learning Theoretical Frameworks
This repository contains code and a research paper exploring the bias-variance trade-off and support vector machines (SVMs).
Repository Structure

- **paper/:**  LaTeX source and compiled PDF of the research paper.
- **bias_variance/:** Python scripts for bias-variance analysis.
- **svm/:** Python scripts for SVM experiments and custom implementation.

Requirements
Install dependencies via:
```
pip install -r requirements.txt
```

Running Experiments

Bias-Variance Analysis:
```
python bias_variance/experiments.py
```
Generates a plot saved to 
```bv_plot.png```

SVM Evaluation:
```
python svm/evaluation.py
```

Prints kernel comparison results and custom SVM accuracy.


Paper Compilation
Compile main.tex using:
```
latexmk -pdf paper/main.tex
```