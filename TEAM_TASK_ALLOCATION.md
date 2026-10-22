# AI-Based Industrial Maintenance and Process Optimization System

## Team Project — Infosys Springboard Internship 7.0

**Showcase target:** Monday, 26 October 2026  
**Current milestone:** Abstract, presentation, and an initial working code demonstration

---

## 1. Project Overview

Industrial machines can fail unexpectedly, causing downtime, repair costs, and production disruption. This project explores how data analysis and machine learning can help identify patterns associated with machine failures and support predictive maintenance.

For this milestone, the team will inspect the dataset, analyze failure patterns, create visualizations, and—if time permits—build a baseline machine-failure prediction model.

> **Scope note:** The current dataset supports machine-failure analysis and predictive-maintenance experiments. Do not claim that production-process optimization has been demonstrated unless the team evaluates relevant production, energy, or cost metrics.

## 2. Team Members and Responsibilities

### Member 1 — Adithyan B (Team Lead and Integration)

**Responsibilities**
- Maintain the shared GitHub repository and project structure.
- Coordinate the team and track progress.
- Prepare the abstract and project overview.
- Prepare slides 1–3: Title, Introduction, and Problem Statement.
- Integrate the code modules and check that the project runs end to end.
- Review the final presentation and submission.

**Deliverables**
- Abstract document
- Slides 1–3
- Integrated project folder and repository
- Final review checklist

### Member 2 — Dheepesh N (Dataset and Preprocessing)

**Responsibilities**
- Prepare slides 4–6: Objectives, Dataset Description, and Dataset Features.
- Create `src/data_preprocessing.py`.
- Load `data/Dataset.csv` using pandas.
- Display the first five rows and dataset shape.
- Inspect column names and data types.
- Check missing values and duplicate rows.
- Report findings without inventing results.

**Deliverables**
- Slides 4–6
- `src/data_preprocessing.py`
- Actual output screenshots or saved output

### Member 3 — Hriday (Failure Analysis and Visualization)

**Responsibilities**
- Prepare slides 7–9: Python Implementation, Data-Quality/Preprocessing Output, and Failure Counts/Visualization.
- Create `src/failure_analysis.py`.
- Count failure and non-failure records using `Machine failure`.
- Calculate their percentages.
- Generate and save a bar chart at `outputs/failure_distribution.png`.
- Explain findings using actual code results.

**Deliverables**
- Slides 7–9
- `src/failure_analysis.py`
- Saved chart and actual output screenshots

### Member 4 — Ganga (Results, Conclusion, and Next Milestone)

**Responsibilities**
- Prepare slides 10–12: Results and Observations, Conclusion, and Next Milestone.
- Collect verified outputs and charts from Members 2 and 3.
- Create `src/predictive_maintenance.py` for a baseline machine-learning model **if time permits**.
- If implementing the model, use suitable input features such as temperature, rotational speed, torque, and tool wear.
- Evaluate the model using precision, recall, F1-score, and a confusion matrix; do not rely on accuracy alone.
- Explain limitations and distinguish completed work from future work.

**Deliverables**
- Slides 10–12
- Results based on actual outputs
- `src/predictive_maintenance.py` and actual evaluation results, if the model is completed

---

## 3. Presentation Plan

The presentation contains 12 slides:

1. **Title Slide** — Project title, team members, guide
2. **Introduction** — Industrial maintenance and AI
3. **Problem Statement** — Unexpected machine failures and downtime
4. **Objectives** — Data preparation and failure analysis
5. **Dataset Description** — Dataset source, size, and columns (verify the source)
6. **Dataset Features** — Input variables and failure labels
7. **Python Implementation 1** — Load the dataset and inspect rows
8. **Python Implementation 2** — Check missing values, duplicates, and data types
9. **Python Implementation 3** — Failure counts and data visualization
10. **Results and Observations** — Actual outputs and charts
11. **Conclusion** — Summary of completed work
12. **Next Milestone** — Machine-failure prediction using machine learning

Keep slide text concise. Use screenshots and charts produced by the team's own code. Do not invent results or describe planned work as completed work.

---

## 4. Suggested GitHub Repository Structure

```text
AI-Industrial-Maintenance-Optimization/
├── data/
│   └── Dataset.csv
├── src/
│   ├── data_preprocessing.py
│   ├── failure_analysis.py
│   └── predictive_maintenance.py
├── outputs/
│   └── failure_distribution.png
├── presentation/
│   └── project_presentation.pptx
├── docs/
│   └── abstract.md
├── requirements.txt
└── README.md
```

Create folders/files as needed. Keep the existing `Dataset.csv` in `data/`. Do not commit private credentials, virtual environments, or unrelated files.

## 5. Initial Coding Requirements

### `src/data_preprocessing.py`
- Read `data/Dataset.csv`.
- Print dataset shape and first five rows.
- Print data types.
- Print missing-value counts.
- Print duplicate-row count.
- Handle file-not-found errors clearly.

### `src/failure_analysis.py`
- Read the dataset.
- Count records where `Machine failure` is 0 and 1.
- Print counts and percentages.
- Save a labelled bar chart to `outputs/failure_distribution.png`.

### `src/predictive_maintenance.py` (optional for this milestone)
- Build a simple, reproducible baseline classifier if time permits.
- Split data into training and testing sets.
- Avoid target leakage: **do not use `TWF`, `HDF`, `PWF`, `OSF`, or `RNF` as input features when predicting `Machine failure`**.
- Report precision, recall, F1-score, and confusion matrix.
- State clearly that this is an initial experiment, not a production-ready system.

## 6. Dependencies

Add the packages needed by the code to `requirements.txt`. A likely starting point is:

```text
pandas
matplotlib
scikit-learn
```

Use only dependencies required by the committed code.

## 7. How to Run the Code

From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python src/data_preprocessing.py
python src/failure_analysis.py
```

Run the prediction script only if implemented:

```bash
python src/predictive_maintenance.py
```

If the dataset is stored elsewhere, update the file path consistently or move it to `data/Dataset.csv`.

## 8. GitHub Collaboration Rules

1. Each member should work on their assigned files and slides.
2. Pull the latest changes before starting work.
3. Test code locally before committing.
4. Commit and push completed changes.
5. Avoid editing the same file simultaneously; coordinate before changing shared files.
6. Use clear commit messages, for example:
   - `Add dataset preprocessing checks`
   - `Add machine failure analysis chart`
   - `Prepare results and conclusion slides`
7. If a script fails, share the error message rather than claiming it works.
8. Keep presentation claims and numerical results consistent with actual code output.

## 9. Completion Checklist

- [ ] Abstract prepared and reviewed
- [ ] All 12 slides drafted
- [ ] Dataset source verified and credited
- [ ] Preprocessing script runs successfully
- [ ] Failure-analysis script runs successfully
- [ ] Failure-distribution chart saved
- [ ] Results slides use genuine outputs
- [ ] Optional ML model tested, if completed
- [ ] Code and presentation pushed to GitHub
- [ ] Every member can explain their own contribution

## 10. Important Reporting Notes

- Confirm the dataset source and cite it in the presentation/README.
- Report counts, percentages, and evaluation metrics from actual script output.
- Machine-failure datasets may be imbalanced; a high accuracy score alone can be misleading.
- Do not use failure subtype columns as model inputs when predicting the overall `Machine failure` label.
- This milestone is an initial prototype. Clearly identify what is implemented now and what is planned for the next milestone.
