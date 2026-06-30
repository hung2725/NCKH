# Conformal Risk Control for Clinical Metrics in Medical Image Segmentation

## Background & Motivation

Modern segmentation models (nnU-Net, SwinUNETR, etc.) produce high-quality masks of organs and lesions, but they come with **no statistical guarantee** about reliability. In practice, clinical decisions rarely rely on pixel-level masks themselves; they rely on **metrics derived from those masks**: left-ventricle volume (heart failure assessment), tumor volume (treatment response monitoring), pulmonary nodule diameter (cancer risk stratification), and so on.

Conformal prediction is a framework that produces prediction intervals with **distribution-free coverage guarantees**. However, most existing work only provides guarantees at the **pixel level** (pixel-level prediction sets). **The gap:** statistical guarantees for **downstream clinical metrics** — the quantities clinicians actually use — remain largely unexplored.

## Research Questions

1. How can we construct prediction intervals for clinical metrics (volume, diameter) with distribution-free coverage guarantees?
2. How can we ensure **group-conditional coverage** (per-organ, per-scanner) rather than only marginal coverage over the whole population?
3. Do the intervals produced by the model correlate with the level of disagreement among annotators (the genuine ambiguity of a case)?

## Proposed Method

- **Foundation step:** use a **pretrained** segmentation model (nnU-Net) — *no retraining required*. From the mask, compute the clinical metric (e.g., volume = number of voxels × voxel volume).
- **Core contribution:** design a **non-conformity score tied directly to the error of the clinical metric**, rather than pixel-level error. Calibrate on a small calibration set to obtain intervals with coverage guarantees.
- **Extension (for Q1-level novelty):** calibrate in the **feature space** — perturbing intermediate features along the low-dimensional subspaces most sensitive to the target metric, yielding tighter intervals (along the lines of COMPASS, ICLR 2026). Analyze **conditional coverage** per organ/scanner.

## Datasets (public, easily accessible)

| Dataset | Clinical metric | Notes |
|---|---|---|
| **ACDC** | Left-ventricle / myocardium volume | Cardiac MRI, standard benchmark |
| **LiTS** | Liver / tumor volume | CT |
| **KiTS** | Kidney tumor volume | CT |
| **LIDC-IDRI** | Pulmonary nodule diameter | **Multiple annotators** → measure ambiguity |

## Baselines

Split conformal, Conformal Risk Control (Angelopoulos et al., ICLR 2024), morphological prediction sets (MICCAI 2025), MC-Dropout, Deep Ensembles.

## Evaluation Metrics

- **Empirical coverage** vs. the target level (e.g., 90%) — is the guarantee actually held?
- **Interval width / efficiency** — narrower is better (at the same coverage).
- **Conditional coverage gap** — difference in coverage across groups.
- **Correlation** between interval width and annotator disagreement.

## Key References

1. Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* (introductory tutorial).
2. Angelopoulos et al. — *Conformal Risk Control* (ICLR 2024).
3. COMPASS — *Robust Feature Conformal Prediction for Medical Segmentation Metrics* (arXiv:2509.22240, ICLR 2026).
