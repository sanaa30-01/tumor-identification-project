# 01 — EDA Findings

**Dataset:** `masoudnickparvar/brain-tumor-mri-dataset`, version 2, pulled via `kagglehub`
**Notebook:** `notebooks/01_load_and_verify.ipynb`
**Date:** August 2026

---

## Summary

The dataset is usable, but the class labels are substantially confounded with
source and acquisition differences. A logistic regression on background pixels
alone — containing no brain tissue — reaches 63.4% accuracy against a 26.2%
majority baseline. The `notumor` class appears to be a different MRI pulse
sequence from the three tumor classes, which means "tumor vs no tumor" on this
dataset is closer to a sequence-recognition task than a diagnostic one.

The meaningful classification problem here is tumor *type*, and specifically
glioma vs meningioma, which is the only pair with no available shortcut.

---



## 1. Dataset composition

- **7,200 images total**, pre-split into `Training/` and `Testing/`
- Four classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Class folders match across both splits
- Roughly balanced — **majority-class baseline = 0.262**

Note: the dataset description cites ~7,023 images. Version 2 contains 7,200.
Report the observed count, not the documented one.

**Provenance:** assembled from three sources (figshare, SARTAJ, Br35H). The
uploader has stated that glioma images from the SARTAJ portion were mislabeled
and replaced with figshare images. This multi-source origin is directly
relevant to Section 3 below.

---



## 2. Image properties


| Property      | Observed                | Implication                                  |
| ------------- | ----------------------- | -------------------------------------------- |
| Width         | 150–1338 px, median 512 | Resize required before batching              |
| Height        | 168–1304 px, median 512 | Same                                         |
| Square        | 79.6% of sampled images | ~1 in 5 needs padding, not stretching        |
| PIL modes     | RGB, L, and one RGBA    | Must normalize to one mode or batching fails |
| Dtype / range | uint8, 0–255            | Divide by 255 during normalization           |


**Aspect ratio decision:** pad to square before resizing rather than stretching.
Stretching distorts anatomy for 20% of the data and risks the model learning
aspect ratio as a shortcut feature.

**Channel decision:** convert everything to a single mode early in the pipeline.
Of 30 sampled RGB images, 28 had identical channels — grayscale stored as RGB.
The remaining ones are covered in Section 3.

---



## 3. Source leakage

This is the most important finding in the EDA.

### 3.1 Colored images cluster entirely in one class

128 of 7,200 images have genuinely colored (non-identical) channels. **All 128
are in** `notumor` — 90 train, 38 test. Zero in any tumor class. Several show
watermark bars and web-source text along the image edge.

A perfect correlation between an artifact and a label is a shortcut feature: a
model can learn to detect the watermark rather than the pathology.

### 3.2 Classes are separable without any anatomy visible

Logistic regression on 8×8 downsampled grayscale images:

```
8×8 accuracy:      0.650
majority baseline: 0.262
```

At 8×8 no tumor is resolvable — 64 blurry pixels. 2.5× baseline means something
other than anatomy carries the signal.

### 3.3 The signal is in the background

Logistic regression on the outer 3-pixel frame of a 32×32 image (348 features,
zero brain tissue):

```
border-only accuracy: 0.634
```

This recovers **97% of the full 8×8 result** from pure background. There is no
legitimate diagnostic information in background pixels — this is watermarks,
padding, black-level differences, and compression signature. Source identity,
encoded in empty space.

### 3.4 Per-class breakdown

Recall from border pixels alone:


| Class      | Recall | Reading             |
| ---------- | ------ | ------------------- |
| notumor    | 0.77   | Strongly leaked     |
| pituitary  | 0.75   | Strongly leaked     |
| glioma     | 0.68   | Leaked              |
| meningioma | 0.33   | Barely above chance |


56 of 123 meningioma cases were predicted as glioma — more than were correctly
classified. **Glioma and meningioma are not separable from background**, meaning
they share a source and acquisition protocol.

### 3.5 Visual confirmation

The sample grid confirms and extends the statistical findings:

- `notumor` **is a different pulse sequence.** The three tumor classes appear
T1-weighted post-contrast: darker overall, bright enhancing lesions, full head
including neck, face, and sinuses. `notumor` is bright and high-contrast with
inverted tissue relationships (T2 or FLAIR), and the brain fills the frame
with minimal neck or facial anatomy.
- **The watermarked images look no different from the rest of their row.** The
colored channels were a marker of the source split, not the confound itself.
- **Glioma and meningioma are visually interchangeable** — same sequence, same
framing, same anatomy on display.
- **Pituitary shares the tumor-class acquisition** but is framed lower, with
orbits and skull base visible in nearly every image. This is the field-of-view
difference the border test detected.



### 3.6 What can and cannot be fixed

**Cropping will help the border leak.** Margin removal deletes exactly the
pixels carrying the background signature — watermark bars, padding, differing
black levels.

**Cropping will not fix the** `notumor` **confound.** A different pulse sequence
changes tissue contrast *inside* the brain. Per-image intensity normalization
won't fully fix it either, since sequence differences alter *relative* contrast
between tissues rather than overall brightness.

There is no fix for this without different data. The response is to measure it,
bound it, and report it.

---



## 4. Decisions carried into preprocessing

1. Pad to square, then resize — do not stretch
2. Convert all images to a single channel mode before batching
3. Divide by 255; compute normalization statistics from the training split only
4. Treat margin removal as leakage mitigation, and **re-run the 8×8 and
  border-only tests on cropped images** to verify it worked
5. The crop finds the largest contour, which in sagittal views is the entire
  head including face and neck — not the brain. This is margin removal, not
   skull-stripping. Skull-stripping is out of scope.

---



## 5. Evaluation plan

**The real baseline is 0.650, not 0.262.** Report CNN accuracy against the
no-anatomy control, not against random guessing.

Run three evaluations:


| Task                       | Purpose                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| 4-class                    | Headline number, comparable to published results on this dataset |
| 3-class (tumor types only) | Removes the `notumor` sequence confound                          |
| Glioma vs meningioma       | The only pair with no shortcut — the real test                   |


**Diagnostic to watch:** glioma/meningioma recall in the final confusion matrix.
High overall accuracy with persistent confusion between those two indicates the
model is riding artifacts on the easy three classes.

Control table to complete:


| Control                     | Accuracy     |
| --------------------------- | ------------ |
| Majority class              | 0.262        |
| 8×8, no anatomy visible     | 0.650        |
| Border pixels only          | 0.634        |
| 8×8 after cropping          | *to measure* |
| Border only, after cropping | *to measure* |


---



## 6. Limitations for the write-up

- **Class labels are confounded with source dataset and pulse sequence.**
Background-only classification reaches 0.634 vs a 0.262 baseline. The
`notumor` class appears to be a different acquisition entirely.
- **No patient IDs.** Slices from the same patient may fall in both train and
validation, inflating validation scores. Cannot be fully resolved without
patient metadata.
- **Single 2D slices, not volumes.** No 3D context available.
- **Assembled from three sources** with heterogeneous acquisition parameters,
and a documented labeling correction in one of them.
- **No segmentation masks**, so no localization and no Dice/IoU.
- The dataset as a whole has a duplication percentage of 44.6%
- The no tumor class has only n = 47 non-duplicate images, indicating that ~88% of  
data in this class was duplicated (in test and train sets) and would have inflated model accuracy

---



## 7. Outstanding — do before splitting

**Near-duplicate detection (Phase 1.5).** Not yet run. Compute a perceptual hash
across the training set and inspect the closest pairs. Because the dataset
exposes no patient IDs, this is the only available estimate of how much
patient-level leakage a random train/val split will introduce.

**Section 6 findings cell.** Fill in the qualitative observations in the
notebook itself while the sample grid is still in view.

---



## Next

`02_preprocessing.ipynb` — contour-based margin removal, pad to square, resize
to 224, normalize. Plot before/after on 20 real images before trusting the crop
function. Then re-run the leakage controls from Section 3 on the cropped data.