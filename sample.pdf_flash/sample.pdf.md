# Advanced Multimodal Systems

Executive overview / mixed columns / raster figure

## ABSTRACT

This five-page synthetic dossier intentionally combines difficult PDF structures: nested headers, rotated labels, embedded raster imagery, vector plots, and multi-column narratives. All measurements are illustrative rather than empirical.

## 1. Research objectives

The benchmark assesses document understanding across heterogeneous visual and textual modalities.

Primary challenges include: - Reading-order reconstruction - Table-cell association - Figure-caption grounding - Metadata preservation

## 2. Pipeline assumptions

Input documents may include scans, vector artwork, nested tables, and nonuniform reading directions.

Expected outputs: - Structured JSON records - Linked figure descriptions - Chunk-level provenance - Layout-aware Markdown

FIGURE 1. Embedded raster microscopy-style illustration

![](img_0001)

Figure 1. Synthetic pixel image with an annotated region of interest (ROI).

Document properties: five distinct page layouts; searchable vector text;

one true raster image; vector diagrams and quantitative visualizations.

**Experimental Matrix**

Multi-level headers / merged regions / dense numerical table

**TABLE 1. Hierarchical performance matrix**

**Method**

**Quality**

**Latency**

**Cost**

**Parser**

**Text F1**

**Table F1**

**Image F1**

**P50 ms**

**P95 ms**

**USD/1k**

Rule-only

0.91

0.62

0.34

82

215

0.08

OCR hybrid

0.94

0.79

0.61

134

388

0.41

Layout model

0.96

0.88

0.83

296

720

0.77

VLM assisted

0.97

0.92

0.95

890

1810

2.65

Ensemble

0.98

0.95

0.96

1020

2300

3.12

Fallback

0.88

0.57

0.29

64

190

0.05

## TABLE 2. Irregular cells with wrapped conceptual content

Condition / group

Observed artifact

Mitigation

A: Multi-column

Cross-column text jumps

Geometry sort

B: Spanning cells

Header-child ambiguity

Cell graph

C: Rotated labels

Orientation mismatch

Angle detect

D: Embedded figure

Caption detached

Region linking

Note: All benchmark values are invented for parser evaluation.

**Architecture & Connectivity**

Vector process diagram / connectors / annotations / legends

## FIGURE 2. Multi-stage document processing graph

## PDF INPUT

raster + vector

## RETRIEVE

semantic query

**SEGMENT** layout zones

**INDEX**

hybrid search

**EXTRACT** text / images

**NORMALIZE** tables / JSON

**Vector symbols and parsing requirements**

## 01 Directed arrows encode dependency

## 02 Each node contains hierarchical text

## 03 Spatial adjacency differs from reading order

04 Boxes and connectors are vector graphics

**Expected structural relation:**

![](img_0002)

**Quantitative Figures**

Line plot / grouped bars / axis labels / embedded legend

FIGURE 3. Quality and throughput across document complexity

![](img_0003)

<details>
<summary>image content</summary>

100
80
60
40
20
0
Low        Moderate        High        Very high        Extreme
Text fidelity        Table fidelity
</details>

FIGURE 4. Processing time by pipeline stage

![](img_0004)

<details>
<summary>image content</summary>

Detection        OCR        Parsing        Embedding
Standard        Complex layout
</details>

Figure 3-4. Synthetic values. Vector-drawn plots are intentionally not bitmap charts.

**Appendix: Edge Cases** Rotated text / two-column reading order / checklist / mixed typography

## A. Extraction validation checklist

Detect merged header cells and infer parent-child relationships

Preserve table column order, units, and numeric precision

Associate captions with nearby figures and raster assets

Recover multi-column paragraphs without interleaving lines

Capture arrows and nodes as a structured directed graph

Retain page number, bounding box, and source provenance

## B. Rotated annotation and dense layout

Column A - embedded narrative A layout-aware parser should retain this contiguous block despite nearby orthogonal text and graphics.

## C. Structured output example

{

## "page": 5, "type": "mixed\_layout",

Column B - metadata Document ID: DEMO-005 Revision: 2.1 (synthetic) Confidence: 0.973 Reading order: A then B

"regions": ["checklist", "rotated\_label", "columns"],

"table\_count": 0, "figure\_count": 0, "requires\_geometry": true

}