// Build paper.docx (RARS-style: US Letter, 1" margins, Arial 12 pt, single-spaced)
// from paper.md. Usage: node build.js <paper.md> <figures-dir> <out.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, Footer,
  PageNumber, LevelFormat, TableLayoutType,
} = require("docx");

const [, , mdPath, figDir, outPath] = process.argv;
const lines = fs.readFileSync(mdPath, "utf8").split("\n");
const FONT = "Arial";
const TEXT_W = 9360; // 6.5 in in DXA

const meta = {};
const children = [];

// ---------- inline formatting: **bold**, *italic*, ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function para(text, opts = {}) {
  return new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120 }, ...opts.p });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const FIG_WIDTH_IN = {
  "fig1_pipeline.png": 6.5, "fig2_rig.png": 6.5, "fig3_false_belief_per_trial.png": 6.2,
  "fig4_detector_fp.png": 5.2, "fig5_qualitative.png": 6.5, "fig6_decay_tradeoff.png": 4.2,
  "fig7_latency.png": 5.4,
};

function figure(caption, file) {
  const base = path.basename(file);
  const fp = path.join(figDir, base);
  const { w, h } = pngSize(fp);
  const widthPx = Math.round((FIG_WIDTH_IN[base] || 6.0) * 96);
  const heightPx = Math.round(widthPx * h / w);
  const m = caption.match(/^(Figure \d+\.)\s*(.*)$/);
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(fp),
      transformation: { width: widthPx, height: heightPx },
      altText: { title: m ? m[1] : "Figure", description: m ? m[2] : caption, name: base },
    })],
  }));
  children.push(new Paragraph({
    spacing: { after: 200 },
    children: [new TextRun({ text: m ? m[1] + " " : "", bold: true, size: 20 }),
               ...runs(m ? m[2] : caption, { size: 20 })],
  }));
}

function table(rowsText, captionText) {
  const rows = rowsText.filter(r => !/^\|\s*-+/.test(r)).map(r =>
    r.replace(/^\|/, "").replace(/\|\s*$/, "").split("|").map(c => c.trim()));
  const ncol = rows[0].length;
  // column widths proportional to max text length (bounded)
  const lens = Array.from({ length: ncol }, (_, j) => {
    const longestWord = Math.max(...rows.map(r => Math.max(...(r[j] || "").replace(/\*\*/g, "").split(/\s+/).map(w => w.length))));
    const full = Math.max(...rows.map(r => (r[j] || "").length));
    return Math.max(longestWord + 3, Math.min(34, full));
  });
  const tot = lens.reduce((a, b) => a + b, 0);
  let widths = lens.map(l => Math.floor(TEXT_W * l / tot));
  widths[ncol - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B9B8B3" };
  const borders = { top: border, bottom: border, left: border, right: border };
  if (captionText) {
    const m = captionText.match(/^(Table \d+\.)\s*(.*)$/);
    children.push(new Paragraph({
      spacing: { before: 160, after: 80 }, keepNext: true,
      children: [new TextRun({ text: m[1] + " ", bold: true, size: 20 }), ...runs(m[2], { size: 20 })],
    }));
  }
  children.push(new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    layout: TableLayoutType.FIXED,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA }, borders,
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "EDECE8" } : undefined,
        children: [new Paragraph({ children: runs(c, { size: 19, bold: i === 0 ? true : undefined }) })],
      })),
    })),
  }));
  children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

// ---------- parse
let i = 0;
let pendingCaption = null;
while (i < lines.length) {
  const line = lines[i];
  if (line.startsWith("% ")) {
    const m = line.match(/^% (\w+): (.*)$/);
    if (m) meta[m[1]] = m[2];
    i++; continue;
  }
  if (line.trim() === "") { i++; continue; }
  if (/^Table \d+\./.test(line) && lines[i + 2] && lines[i + 2].startsWith("|")) {
    pendingCaption = line; i++; continue;
  }
  if (line.startsWith("|")) {
    const block = [];
    while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
    table(block, pendingCaption); pendingCaption = null; continue;
  }
  let m;
  if ((m = line.match(/^!\[(.*)\]\((.*)\)$/))) { figure(m[1], m[2]); i++; continue; }
  if (line.startsWith("## ")) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
      children: [new TextRun({ text: line.slice(3) })] }));
    i++; continue;
  }
  if (line.startsWith("# ")) {
    const t = line.slice(2);
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true,
      children: [new TextRun({ text: t })] }));
    i++; continue;
  }
  if (line.startsWith("> ")) {
    children.push(new Paragraph({ indent: { left: 720, right: 720 }, spacing: { after: 160 },
      children: runs(line.slice(2), { italics: true }) }));
    i++; continue;
  }
  if ((m = line.match(/^(\d+)\. (.*)$/))) {
    children.push(new Paragraph({ numbering: { reference: "nums", level: 0 },
      spacing: { after: 80 }, children: runs(m[2]) }));
    i++; continue;
  }
  if (line.startsWith("- ")) {
    children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
      spacing: { after: 80 }, children: runs(line.slice(2)) }));
    i++; continue;
  }
  if (/^\[\d+\] /.test(line)) {
    children.push(new Paragraph({ indent: { left: 540, hanging: 540 }, spacing: { after: 100 },
      children: runs(line, { size: 21 }) }));
    i++; continue;
  }
  if (line.startsWith("python ")) {
    children.push(new Paragraph({ spacing: { after: 120 }, indent: { left: 360 },
      children: [new TextRun({ text: line, font: "Courier New", size: 18 })] }));
    i++; continue;
  }
  children.push(para(line));
  i++;
}

// ---------- title block
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: meta.TITLE, bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: meta.AUTHOR, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: meta.AFFIL, size: 22, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: meta.DATE, size: 22, color: "52514E" })] }),
];

const doc = new Document({
  creator: meta.AUTHOR, title: meta.TITLE,
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true },
        paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: [...title, ...children],
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });
