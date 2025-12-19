# Mako Shark Growth Data - Sources and Verification Guide

## Overview

The file `mako_growth_data_verified.csv` contains age-length-weight data for shortfin mako sharks (*Isurus oxyrinchus*) derived from **validated growth parameters** published in peer-reviewed scientific literature.

## How the Data Was Generated

The CSV data was **NOT** collected directly from individual sharks. Instead, it contains **predicted length-at-age values** calculated using published von Bertalanffy and Gompertz growth model parameters. This is a standard approach in fisheries science for generating growth curves for model fitting.

### Growth Equations Used

**Von Bertalanffy Growth Function (3-parameter):**
```
L(t) = L∞ - (L∞ - L₀) × exp(-K × t)
```

**Gompertz Growth Function (3-parameter):**
```
L(t) = L∞ × exp(ln(L₀/L∞) × exp(-K × t))
```

Where:
- `L(t)` = Fork length at age t (cm)
- `L∞` = Asymptotic maximum length (cm)
- `K` = Growth rate coefficient (year⁻¹)
- `L₀` = Length at birth/age 0 (cm)
- `t` = Age (years)

### Length-to-Weight Conversion

Weight was calculated using the length-weight relationship from Kohler et al. (1996):
```
W(kg) = 4.4 × 10⁻⁶ × FL(cm)^3.14
```

---

## Primary Source Documents

All source PDFs are included in this repository. You can verify the parameters by opening the PDFs and checking the specified pages/tables.

### 1. Natanson et al. (2006) - Western North Atlantic

**PDF File:** `Validated_age_and_growth_estimates_for_the_shortfi.pdf`

**Full Citation:**
> Natanson LJ, Kohler NE, Ardizzone D, Cailliet GM, Wintner SP, Mollet HF (2006) Validated Age and Growth Estimates for the Shortfin Mako, *Isurus oxyrinchus*, in the North Atlantic Ocean. *Environmental Biology of Fishes* 77:367-383

**Where to Find Parameters:**
- **Table 1, Page 10** of the PDF
- Look for "von Bertalanffy—3 parameter" and "Gompertz—3 parameter" rows

**Parameters Used:**

| Sex | Model | L∞ (cm FL) | K (year⁻¹) | L₀ (cm) | Sample Size |
|-----|-------|-----------|------------|---------|-------------|
| Male | von Bertalanffy | 253.3 | 0.125 | 71.6 | n=118 |
| Female | Gompertz | 365.6 | 0.087 | 88.4 | n=140 |

**Validation Method:** Bomb radiocarbon dating and oxytetracycline (OTC) injection

---

### 2. Semba et al. (2009) - Western/Central North Pacific

**PDF File:** `Age_and_growth_analysis_of_the_shortfin_mako_Isuru.pdf`

**Full Citation:**
> Semba Y, Nakano H, Aoki I (2009) Age and growth analysis of the shortfin mako, *Isurus oxyrinchus*, in the western and central North Pacific Ocean. *Environmental Biology of Fishes* 84:377-391

**Where to Find Parameters:**
- **Table 2, Page 9** of the PDF (page 385 of the journal)
- Look for the row "Males" and "Females" with L∞, K, and L₀ columns

**Parameters Used:**

| Sex | Model | L∞ (cm PCL) | K (year⁻¹) | L₀ (cm) | Sample Size |
|-----|-------|-------------|------------|---------|-------------|
| Male | von Bertalanffy | 231.0 | 0.16 | 59.7 | n=128 |
| Female | von Bertalanffy | 308.3 | 0.090 | 59.7 | n=147 |

**Note:** PCL (Precaudal Length) was converted to FL (Fork Length) using:
```
FL = 1.10 × PCL + 8.1
```
(Conversion from Bishop 2004, as cited in the paper)

---

### 3. Groeneveld et al. (2014) - Southwest Indian Ocean

**PDF File:** `2014_Makobiology_MFR.pdf`

**Full Citation:**
> Groeneveld JC, Cliff G, Dudley SFJ, et al. (2014) Population structure and biology of shortfin mako, *Isurus oxyrinchus*, in the SW Indian Ocean. *Marine and Freshwater Research* 65:1045-1058

**Where to Find Parameters:**
- **Page 8** of the PDF (page 1050 of the journal)
- Look in the text for: "growth parameters (with 95% confidence intervals) were estimated to be 90.4cm (79.6–101.0cm) for L₀, 285.4cm (237.1–333.7cm) for L∞ and 0.113 y⁻¹ (0.058–0.168 y⁻¹) for k"

**Parameters Used:**

| Sex | Model | L∞ (cm FL) | K (year⁻¹) | L₀ (cm) | Sample Size |
|-----|-------|-----------|------------|---------|-------------|
| Combined | von Bertalanffy | 285.4 | 0.113 | 90.4 | n=89 |

---

## Verification Checklist

To verify this data is not fabricated:

- [ ] Open `Validated_age_and_growth_estimates_for_the_shortfi.pdf`, go to Table 1 on page 10, and confirm L∞, K, and L₀ values for males and females
- [ ] Open `Age_and_growth_analysis_of_the_shortfin_mako_Isuru.pdf`, go to Table 2 on page 9, and confirm the growth parameters
- [ ] Open `2014_Makobiology_MFR.pdf`, read page 8, and find the text with growth parameter values
- [ ] Use the growth equations above with the published parameters to calculate length at any age

## Using This Data for Model Fitting

This data is suitable for:
1. Fitting von Bertalanffy or Gompertz growth models
2. Comparing growth rates across different ocean populations
3. Estimating mass-at-age relationships
4. Teaching/demonstrating differential equation models in biology

**Recommended Approach:**
- For model fitting, consider using data from a single source/region for consistency
- The Natanson et al. (2006) data is particularly reliable as it uses validated aging methods

---

## Additional References

### Length-Weight Relationship

> Kohler NE, Casey JG, Turner PA (1996) Length-length and length-weight relationships for 13 species of sharks from the western North Atlantic. NOAA Technical Memorandum NMFS-NE-110

### PCL to FL Conversion

> Bishop SDH (2004) Age determination and life history characteristics of the shortfin mako (*Isurus oxyrinchus*) in New Zealand waters. Master's Thesis, University of Auckland, New Zealand
