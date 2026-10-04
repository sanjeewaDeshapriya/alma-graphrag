# Unweighted vs weighted GraphRAG (77-hotel Colombo pool)

## Main benchmark (60 queries)

| System | P@10 | R@10 | nDCG@10 | MRR | Top-1 |
|---|---|---|---|---|---|
| Random | 0.235 | 0.134 | 0.202 | 0.436 | 0.217 |
| Popularity | 0.190 | 0.114 | 0.189 | 0.195 | 0.167 |
| Filter | 0.412 | 0.229 | 0.501 | 0.543 | 0.533 |
| Keyword | 0.488 | 0.344 | 0.526 | 0.763 | 0.683 |
| SemanticVec | 0.297 | 0.192 | 0.262 | 0.514 | 0.333 |
| Hybrid | 0.422 | 0.290 | 0.445 | 0.771 | 0.700 |
| CrossEncoder | 0.403 | 0.243 | 0.373 | 0.684 | 0.567 |
| LTR | 0.915 | 0.597 | 0.958 | 0.967 | 0.967 |
| LTR[doc-only] | 0.535 | 0.328 | 0.505 | 0.762 | 0.717 |
| GraphRAG[unweighted] | 0.710 | 0.439 | 0.767 | 0.927 | 0.917 |
| WeightedGraphRAG | 0.728 | 0.473 | 0.812 | 0.937 | 0.917 |
| GraphRAG[unweighted,no-diffusion] | 0.710 | 0.439 | 0.767 | 0.927 | 0.917 |

| Unweighted vs | W/T/L | d_z | Relative gain | Diff [95% CI], Holm p |
|---|---|---|---|---|
| Filter | 28/32/0 | 0.85 | +53.1% | +0.266 [+0.188, +0.346], p=0.0 |
| Keyword | 49/2/9 | 0.73 | +45.9% | +0.241 [+0.157, +0.324], p=0.0 |
| SemanticVec | 57/0/3 | 1.64 | +193.4% | +0.506 [+0.426, +0.581], p=0.0 |
| Hybrid | 53/1/6 | 1.14 | +72.5% | +0.322 [+0.249, +0.392], p=0.0 |
| CrossEncoder | 55/0/5 | 1.25 | +105.9% | +0.395 [+0.313, +0.473], p=0.0 |
| WeightedGraphRAG | 0/35/25 | -0.49 | -5.5% | -0.045 [-0.069, -0.024], p=0.0 |

Weighted vs unweighted, per query: 25 wins / 35 ties / 0 losses (d_z 0.49).

| Category | Filter | Keyword | Unweighted | Weighted |
|---|---|---|---|---|
| accessibility | 0.000 | 0.320 | 0.405 | 0.588 |
| amenity | 0.500 | 0.768 | 0.786 | 0.788 |
| disruption | 0.000 | 0.400 | 0.724 | 0.808 |
| economic | 0.830 | 0.515 | 0.830 | 0.830 |
| multi_dimensional | 0.719 | 0.443 | 0.900 | 0.900 |
| quality | 0.958 | 0.708 | 0.958 | 0.958 |

## Price slice (20 queries)

| System | P@10 | R@10 | nDCG@10 | MRR | Top-1 |
|---|---|---|---|---|---|
| Random | 0.350 | 0.111 | 0.259 | 0.552 | 0.350 |
| Popularity | 0.250 | 0.079 | 0.173 | 0.125 | 0.000 |
| Filter | 0.310 | 0.098 | 0.236 | 0.212 | 0.050 |
| Keyword | 0.380 | 0.124 | 0.298 | 0.591 | 0.400 |
| SemanticVec | 0.415 | 0.131 | 0.369 | 0.573 | 0.350 |
| Hybrid | 0.420 | 0.136 | 0.368 | 0.637 | 0.450 |
| CrossEncoder | 0.395 | 0.125 | 0.265 | 0.413 | 0.200 |
| LTR | 0.915 | 0.292 | 0.893 | 0.950 | 0.950 |
| LTR[doc-only] | 0.325 | 0.109 | 0.283 | 0.327 | 0.250 |
| GraphRAG[unweighted] | 0.765 | 0.247 | 0.780 | 1.000 | 1.000 |
| WeightedGraphRAG | 0.775 | 0.248 | 0.763 | 0.950 | 0.900 |
| GraphRAG[unweighted,no-diffusion] | 0.765 | 0.247 | 0.780 | 1.000 | 1.000 |

| Unweighted vs | W/T/L | d_z | Relative gain | Diff [95% CI], Holm p |
|---|---|---|---|---|
| Filter | 20/0/0 | 2.55 | +230.2% | +0.544 [+0.447, +0.634], p=0.0005 |
| Keyword | 19/0/1 | 1.96 | +161.7% | +0.482 [+0.373, +0.586], p=0.0 |
| SemanticVec | 20/0/0 | 2.18 | +111.2% | +0.410 [+0.325, +0.494], p=0.0 |
| Hybrid | 19/0/1 | 2.08 | +111.8% | +0.412 [+0.323, +0.497], p=0.0 |
| CrossEncoder | 20/0/0 | 2.81 | +194.1% | +0.515 [+0.433, +0.596], p=0.0 |
| WeightedGraphRAG | 12/0/8 | 0.18 | +2.2% | +0.017 [-0.022, +0.061], p=1.0 |

Weighted vs unweighted, per query: 8 wins / 0 ties / 12 losses (d_z -0.18).

| Category | Filter | Keyword | Unweighted | Weighted |
|---|---|---|---|---|
| economic_ranked | 0.236 | 0.298 | 0.780 | 0.763 |

## Ablations (nDCG@10)

**unweighted, main:** Full 0.767; w/o spatial 0.720 (-0.047*); w/o accessibility 0.716 (-0.052*); w/o facility 0.736 (-0.032); w/o economic 0.856 (+0.089*); w/o disruption 0.759 (-0.008); w/o feasibility filter 0.494 (-0.273*); w/o neighbourhood diffusion 0.767 (-0.000); only spatial 0.741 (-0.026); only accessibility 0.884 (+0.117); only facility 0.650 (-0.117*); only economic 0.608 (-0.159*); only disruption 0.667 (-0.101*); Filter 0.501 (-0.266*)

**unweighted, price:** Full 0.782; w/o spatial 0.843 (+0.061*); w/o accessibility 0.871 (+0.089*); w/o facility 0.763 (-0.019); w/o economic 0.378 (-0.404*); w/o disruption 0.789 (+0.008); w/o feasibility filter 0.729 (-0.053); w/o neighbourhood diffusion 0.782 (-0.000); only spatial 0.373 (-0.408*); only accessibility 0.358 (-0.424*); only facility 0.372 (-0.410*); only economic 0.975 (+0.193*); only disruption 0.328 (-0.454*); Filter 0.236 (-0.546*)

**weighted, main:** Full 0.812; w/o spatial 0.798 (-0.014); w/o accessibility 0.717 (-0.095*); w/o facility 0.776 (-0.036*); w/o economic 0.866 (+0.054*); w/o disruption 0.815 (+0.004); w/o intent ladder 0.787 (-0.025*); w/o feasibility filter 0.564 (-0.248*); w/o neighbourhood diffusion 0.812 (-0.000); only spatial 0.741 (-0.071*); only accessibility 0.884 (+0.072); only facility 0.650 (-0.162*); only economic 0.608 (-0.204*); only disruption 0.667 (-0.145*); Filter 0.501 (-0.311*)
  simplex: {'median': 0.7921, 'share_above_filter': 1.0, 'handset_percentile': 0.6}

**weighted, price:** Full 0.781; w/o spatial 0.858 (+0.076*); w/o accessibility 0.868 (+0.087*); w/o facility 0.767 (-0.014); w/o economic 0.383 (-0.399*); w/o disruption 0.788 (+0.007); w/o intent ladder 0.716 (-0.065*); w/o feasibility filter 0.725 (-0.056); w/o neighbourhood diffusion 0.781 (-0.000); only spatial 0.373 (-0.408*); only accessibility 0.358 (-0.424*); only facility 0.372 (-0.409*); only economic 0.975 (+0.194*); only disruption 0.328 (-0.454*); Filter 0.236 (-0.545*)
  simplex: {'median': 0.7804, 'share_above_filter': 1.0, 'handset_percentile': 0.504}

## Controlled disruption (49 closures, 0.75 km grid)

**unweighted** (reference GraphRAG[unweighted]): {"coverage=0.5": {"live_minus_stale": 0.0114, "live_vs_stale_p": 0.0006430412466952321, "vs_filter": {"diff": 0.0075, "p_holm": 0.6292760832880053, "significant": false}}, "coverage=1": {"live_minus_stale": 0.0311, "live_vs_stale_p": 0.00043777719457466344, "vs_filter": {"diff": 0.0272, "p_holm": 0.002868153222399379, "significant": true}}}

**weighted** (reference WeightedGraphRAG): {"coverage=0.5": {"live_minus_stale": 0.011, "live_vs_stale_p": 0.004649433819808542, "vs_filter": {"diff": 0.0079, "p_holm": 0.32616559254921484, "significant": false}}, "coverage=1": {"live_minus_stale": 0.0324, "live_vs_stale_p": 0.00043777719457466344, "vs_filter": {"diff": 0.0292, "p_holm": 0.0007188041273771093, "significant": true}}}

## Held-out booking choices

```
{
  "unweighted_split": {
    "Filter": {
      "choice_ndcg": 0.195,
      "hit10": 0.354,
      "mean_rank": 17.18
    },
    "Keyword": {
      "choice_ndcg": 0.171,
      "hit10": 0.396,
      "mean_rank": 16.07
    },
    "SemanticVec": {
      "choice_ndcg": 0.097,
      "hit10": 0.246,
      "mean_rank": 16.54
    },
    "Hybrid": {
      "choice_ndcg": 0.174,
      "hit10": 0.321,
      "mean_rank": 16.58
    },
    "GraphRAG[unweighted]": {
      "choice_ndcg": 0.544,
      "hit10": 0.893,
      "mean_rank": 5.35
    }
  },
  "weighted_handset_split": {
    "choice_ndcg": 0.585,
    "hit10": 0.893,
    "mean_rank": 4.72
  },
  "fifty_splits": {
    "unweighted": {
      "choice_ndcg": 0.5067,
      "sd": 0.0228,
      "mean_rank": 6.2738
    },
    "handset": {
      "choice_ndcg": 0.5471,
      "sd": 0.0243,
      "mean_rank": 5.5486
    },
    "human": {
      "choice_ndcg": 0.5176,
      "sd": 0.0242,
      "mean_rank": 6.2608
    },
    "fitted-on-train": {
      "choice_ndcg": 0.6133,
      "sd": 0.0282,
      "mean_rank": 4.6031
    },
    "unweighted_minus_handset": {
      "mean": -0.0404,
      "sd": 0.0093,
      "p2_5": -0.0574,
      "p97_5": -0.0254,
      "min": -0.0584,
      "max": -0.0251
    }
  }
}
```

`*` = significant after Holm correction. Paired bootstrap, 5,000 resamples.
