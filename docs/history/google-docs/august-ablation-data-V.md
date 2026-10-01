# AVI — August ablation data (Google Doc titled "V")

> Source: Google Doc "V" · (private Google Drive link removed)
> Created 2026-08-15 · Last modified 2026-08-15 · Imported 2026-09-28 (verbatim)
> Raw numbers for the August synthetic + live corpora. Explained in method-metrics-findings.md.

2-DOF camera arm. Arduino Uno + Logitech C270. RTX 3060 Laptop 6GB. YOLO-World detector + Qwen2.5-VL-3B 4-bit (transformers).

## SYNTHETIC ABLATION — 36 episodes, 816 frames, 12 disruption types, 3 repeats
durAc / frames-lost / false-belief

| cond | environment | identity | occlusion | unexpected |
| :- | :- | :- | :- | :- |
| A_vlm_only | 0.972/7/0 | 0.0/26/0 | 0.333/52/0 | 1.0/0/0 |
| B_detector | 0.992/2/0 | 0.0/26/24 | 0.333/52/0 | 1.0/0/0 |
| C_stateless | 0.976/6/0 | 0.0/26/0 | 0.333/52/0 | 1.0/0/0 |
| D_full | 0.992/2/0 | 0.952/2/0 | 0.968/4/0 | 0.691/0/24 |
| E_no_vlm | 0.992/2/0 | 0.952/2/24 | 0.968/4/0 | 0.691/0/24 |
| F_probe | 0.992/2/0 | 0.595/14/0 | 0.786/24/0 | 0.869/0/8 |

## LIVE CORPUS — 10 episodes, 1613 frames, 1099 scored, 514 edge-excluded
| cond | environment | occlusion | unexpected |
| :- | :- | :- | :- |
| D_full | 1.0/0/0 | 0.299/160/5 | 1.0/0/0 |
| E_no_vlm | 1.0/0/0 | 0.293/162/23 | 1.0/0/0 |
| F_probe | 1.0/0/0 | 0.504/98/5 | 1.0/0/0 |

| scenario | occluder | D_full | E_no_vlm | F_probe | winner |
| :- | :- | :- | :- | :- | :- |
| occlusion_full | hand | 0.05/38 | 0.05/38 | 1.00/0 | F |
| occlusion_remove | hand | 0.05/38 | 0.05/38 | 1.00/0 | F |
| occlude_book | book | 0.562/14 | 0.656/11 | 0.562/14 | tie |
| occlude_paper | paper | 0.0/25 | 0.0/25 | 0.0/25 | all fail |
| occlude_box | box | 0.387/19 | 0.161/26 | 0.129/27 | D |
| occlude_jacket | jacket | 0.429/16 | 0.214/22 | 0.143/24 | D |
| identity_same_class | hand then removal | 0.615/10 | 0.923/2 | 0.692/8 | E |
| ALL OCCLUSION | 7 ep | 0.299/160 | 0.293/162 | 0.504/98 | F |

### PER-OCCLUDER (durAcc / lost) — F's aggregate win comes ENTIRELY [text truncated in original doc]

## SYNTHETIC vs LIVE INVERSION
| corpus | D_full | F_probe | higher |
| :- | :- | :- | :- |
| synthetic 36 ep | 0.968/4 | 0.786/24 | D |
| live 7 occlusion ep | 0.299/160 | 0.504/98 | F |

## WHAT THE VLM ACTUALLY CALLS OCCLUDERS — 5 samples each
| real occluder | VLM said | in list | durAcc | generated said | predicted? |
| :- | :- | :- | :- | :- | :- |
| hand | "hands" x5 | yes | 1.000 | "hand" occluder | yes |
| book | "desk lamp" x2, "purple book", "notebook" | mostly no | 0.562 | "book" occluder | NO |
| box | "notebook" x5 | no | 0.129 | "bookend" replacement | yes |
| jacket | "socks" x5 | no | 0.143 | "suit" replacement | yes |
| paper | "document" x5 | no | 0.000 | "paper sheet" occluder | NO |

## DEGRADATION ENVELOPE — 5 real base frames per level
| axis | levels (level: detect-rate/confidence) | breaks at |
| :- | :- | :- |
| darkness (brightness mult) | 1.0:1.00/0.984, 0.7:1.00/0.985, 0.5:1.00/0.985, 0.35:1.00/0.984, 0.25:1.00/0.982, 0.15:1.00/0.955, 0.08:1.00/0.735 | NEVER |
| blur (gaussian px) | 0:1.00/0.984, 2:1.00/0.982, 4:1.00/0.980, 7:1.00/0.978, 11:1.00/0.977, 16:1.00/0.972, 22:1.00/0.971 | NEVER |
| motion blur (streak px) | 1:1.00/0.984, 5:1.00/0.981, 11:1.00/0.975, 21:1.00/0.959, 35:1.00/0.971, 51:1.00/0.946, 71:1.00/0.606 | NEVER |
| noise (sigma) | 0:1.00/0.984, 10:1.00/0.985, 20:1.00/0.978, 35:1.00/0.948, 55:1.00/0.852, 80:1.00/0.628, 110:0.60/0.258 | sigma 110 only |
| downscale (res factor) | 1.0:1.00/0.984, 0.5:1.00/0.984, 0.3:1.00/0.974, 0.2:0.80/0.528, 0.12:0.00/0.051, 0.08:0.00/0.035, 0.05:0.00/0.000 | **0.12x** |
| frame-edge truncation | 0-90% clipped | NEVER |
| local shadow (target only) | to 0.05x on region | NEVER |
| glare (specular blowout) | to full | NEVER |
| cover top / bottom / side | 0-90% | 90% all three |

### COVERAGE with REAL occluder pixels (conf at coverage fraction)
| occluder | 0% | 25% | 50% | 70% | 85% | 100% |
| :- | :- | :- | :- | :- | :- | :- |
| hand | 0.983 | 0.975 | 0.968 | 0.919 | 0.684 | 0.000 |
| book | 0.983 | 0.982 | 0.899 | 0.923 | 0.954 | 0.130 |
| box | 0.981 | 0.970 | 0.926 | 0.922 | 0.345 | 0.000 |
| jacket | 0.983 | 0.978 | 0.901 | 0.793 | 0.000 | 0.000 |

## SEARCH TRIALS (arm actually moving) — PROVISIONAL/SUPERSEDED
| brightness | found | centred | med t-find | med moves |
| :- | :- | :- | :- | :- |
| 1.00 | 2/2 | 2/2 | 1.96s | 0.5 |
| 0.50 | 2/2 | 0/2 | 1.06s | 4.0 |
| 0.25 | 2/2 | 1/2 | 0.52s | 0.0 |
| 0.12 | 1/2 | 0/2 | 5.56s | 6.0 |
| 0.06 | 0/2 | 0/2 | - | 8.0 |

### CAPTURE RATE — the real ceiling
| condition | fps | read failures |
| :- | :- | :- |
| camera alone | 10.1 | 0 |
| after 5s idle | 10.2 | 0 |
| + detector every frame | 8.5 | 0 |
| + detector + VLM every frame | 3.7 | 0 |

### GPU CONTENTION (ComfyUI CPU-mode running alongside)
| state | detector | VLM | per frame |
| :- | :- | :- | :- |
| idle | 14.0 ms | 190.6 ms | 0.205 s |
| generating | 22.1 ms | 308.0 ms | 0.330 s |
| penalty | 1.58x | 1.62x | 1.61x |

## VRAM (measured, corrects an earlier estimate)
| thing | peak VRAM |
| :- | :- |
| card total (RTX 3060 Laptop) | 6144 MiB |
| AVI full stack (YOLO + VLM) | 3802 MiB |
| VLM alone | 2511 MiB (flat at every pixel budget) |
| ComfyUI GPU 512x512 | 2686 MiB, 8.2 s/img |
| ComfyUI GPU 1280x720 | 3238 MiB, 11.3 s/img |
| ComfyUI CPU | 0 MiB, 184 s/img |

## MEASUREMENT BUGS FOUND AND FIXED (9)
| bug | wrong answer it produced | direction |
| :- | :- | :- |
| duplicate crops in corpus | n inflated 12x; identity claim rested on ONE image | overclaim |
| "mug" distractor at IoU 0.995 with target | pasted a real red cup where GT said absent | invalid GT |
| probe tested with logits verifier | "hypothesis disproven" — that verifier cannot answer open questions, scored a hand 0/10 | false negative |
| decay clock overrode probe | identical counts probe on/off; probe looked useless | masked effect |
| stale _probe_at throttle | probe silently dead after the FIRST occlusion of a session, no error | silent |
| cross-run latency drift | nearly reported 4.18x speedup (same config measured 0.644s and 1.671s) | overclaim |
| substring keyword match | "notebook","handbag","armchair","paperweight" all read as OCCLUDERS = belief held on a target that is GONE | DANGEROUS |
| whole-word fix broke plurals | "Hands" no longer matched "hand" — the single most common occlusion case | safe |
| inverted arm re-home sign | arm walked 90 to 150 to 180 (mechanical stop) by trial 2, sat ENERGISED there for 8 trials while log printed plausible rows | SAFETY |
