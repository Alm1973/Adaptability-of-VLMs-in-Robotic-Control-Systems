# Paper

*When Does a Local Vision-Language Model Help? Event-Gated Verification for a Robot Camera Under Scene Disruptions.* Shaurya S. Khidake, BASIS Phoenix. Polygence research project, 2026.

| File | What it is |
|---|---|
| `paper.pdf` | Read this one |
| `paper.docx` | Word version (US Letter, 1-inch margins, Arial 12 pt) for editing and for the final PDF export |
| `paper.md` | Source text |
| `build.js` | Builds `paper.docx` from `paper.md` and the figures |

To rebuild the Word file (Node.js and the `docx` package):

```bash
cd paper
npm install docx
node build.js paper.md ../results/figures paper.docx
```

Every number in the paper comes from `results/analysis/analyze.py`. The pilot results in Section 5.10 come from the August lab notebook and write-up in `docs/`.
