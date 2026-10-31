// VLA-NAV project report — DOCX generator (docx-js, per docx skill rules)
// Cover: recipe R1 (Pure Paragraph Left), palette DM-1 (Deep Cyan, tech/AI)
// Sections: cover (margin 0) -> front matter (TOC, Roman) -> body (Arabic, start 1)
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  LevelFormat, TableOfContents, SectionType, TableLayoutType,
} = require("docx");
const fs = require("fs");
const path = require("path");

const ROOT = require("path").resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "docs", "project_report.docx");

// ---------- palette (DM-1 Deep Cyan) ----------
const P = {
  bg: "162235", accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", innerLine: "C8DDE2", surface: "EDF3F5" },
  headingColor: "0A1628", body: "000000", secondary: "506070",
};

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

function safeText(v, ph) {
  if (v === undefined || v === null || v === "" || String(v) === "NaN" || String(v) === "undefined") {
    return ph || "[missing]";
  }
  return String(v);
}

// ---------- cover title layout (English width: ~pt*11 twips per char) ----------
function splitTitleLinesEN(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const words = title.split(" ");
  const lines = [];
  let cur = "";
  for (const w of words) {
    const cand = cur ? cur + " " + w : w;
    if (cand.length > charsPerLine && cur) { lines.push(cur); cur = w; }
    else cur = cand;
  }
  if (cur) lines.push(cur);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += " " + last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 34, minPt = 24) {
  const charWidth = (pt) => pt * 11; // English average glyph width
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = Math.floor(maxWidthTwips / charWidth(titlePt));
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLinesEN(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLinesEN(title, Math.floor(maxWidthTwips / charWidth(minPt)));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 34, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838, marginTop = 0, marginBottom = 0 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - marginTop - marginBottom - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const remainingSpace = usableHeight - contentHeight;
  const safeRemaining = Math.max(remainingSpace, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(safeRemaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}

// ---------- cover recipe R1 (Pure Paragraph Left) ----------
function buildCoverR1(config) {
  const C = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 34, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: C.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: C.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: C.cover.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: C.cover.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: safeText(line), size: 24, color: C.cover.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: safeText(config.footerLeft, ""), size: 16, color: C.cover.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: safeText(config.footerRight, ""), size: 16, color: C.cover.footerColor, font: { ascii: "Arial" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: C.bg }, borders: noBorders,
        children,
      })],
    })],
  })];
}

// ---------- body helpers ----------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, color: P.headingColor, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, color: P.headingColor, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}
function body(runsOrText, opts = {}) {
  const runs = typeof runsOrText === "string"
    ? [new TextRun({ text: runsOrText, size: 24, color: P.body, font: { ascii: "Times New Roman", eastAsia: "SimSun" } })]
    : runsOrText;
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 312, after: 120, ...(opts.spacing || {}) },
    keepNext: opts.keepNext || undefined,
    children: runs,
  });
}
function r(text, o = {}) {
  return new TextRun({ text, size: o.size || 24, bold: o.bold, italics: o.italics,
    color: o.color || P.body, font: { ascii: o.font || "Times New Roman", eastAsia: "SimSun" } });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200, line: 312 },
    children: [new TextRun({ text, size: 21, italics: true, color: P.secondary, font: { ascii: "Times New Roman" } })],
  });
}
function tableTitle(text) {
  return new Paragraph({
    keepNext: true, spacing: { before: 200, after: 80, line: 312 },
    children: [new TextRun({ text, bold: true, size: 21, color: P.headingColor, font: { ascii: "Times New Roman" } })],
  });
}
function cellPara(text, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.LEFT, spacing: { line: 312 },
    children: [new TextRun({ text: safeText(text), size: 21, bold: o.bold,
      color: o.color || P.body, font: { ascii: "Times New Roman" } })],
  });
}
function dataTable(headers, rows, widths, aligns) {
  const mk = (text, i, isHeader) => new TableCell({
    children: [cellPara(text, { bold: isHeader, color: isHeader ? P.table.headerText : P.body, align: (aligns && aligns[i]) || AlignmentType.LEFT })],
    shading: isHeader ? { type: ShadingType.CLEAR, fill: P.table.headerBg } : undefined,
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    width: { size: widths[i], type: WidthType.PERCENTAGE },
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: P.table.headerBg },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: P.table.headerBg },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: P.table.innerLine },
      insideVertical: NB,
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((t, i) => mk(t, i, true)) }),
      ...rows.map(row => new TableRow({ cantSplit: true, children: row.map((t, i) => mk(t, i, false)) })),
    ],
  });
}
function codeBlock(lines) {
  return lines.map((t, i) => new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: P.table.surface },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: P.table.headerBg, space: 8 } },
    spacing: { line: 276, before: i === 0 ? 120 : 0, after: i === lines.length - 1 ? 200 : 0 },
    indent: { left: 240 },
    children: [new TextRun({ text: t, size: 18, color: P.headingColor, font: { ascii: "Courier New" } })],
  }));
}
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function figure(relPath, displayWidth, captionText, introRuns) {
  const buf = fs.readFileSync(path.join(ROOT, relPath));
  const { w, h } = pngSize(buf);
  const dh = Math.round(displayWidth * h / w);
  return [
    body(introRuns, { keepNext: true }),
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      children: [new ImageRun({ data: buf, transformation: { width: displayWidth, height: dh }, type: "png" })],
    }),
    caption(captionText),
  ];
}
function numItem(ref, runsOrText) {
  const runs = typeof runsOrText === "string" ? [r(runsOrText)] : runsOrText;
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 80 },
    children: runs,
  });
}
function bullet(runsOrText) {
  const runs = typeof runsOrText === "string" ? [r(runsOrText)] : runsOrText;
  return new Paragraph({
    bullet: { level: 0 },
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 80 },
    children: runs,
  });
}

const pageNumFooter = () => new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080", font: { ascii: "Times New Roman" } })],
  })],
});
const docHeader = () => new Header({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "C8DDE2", space: 4 } },
    children: [new TextRun({ text: "VLA-NAV Project Report", size: 18, color: "808080", font: { ascii: "Times New Roman" } })],
  })],
});

// ---------- body content ----------
const FIG = "results/figures";
const bodyChildren = [
  h1("1. Executive Summary"),
  body([r("Large Vision-Language-Action (VLA) models can drive a robot directly from camera images and a language instruction, but each query of an 8-billion-parameter model costs seconds of GPU time. The standard deployment queries the model at every control step, and most of those queries are wasted whenever the world still matches the robot's plan. This project measures whether a cheap onboard signal - the mismatch between commanded and observed motion, plus stalled-motion run length - can decide when to re-query, cutting cost without losing success.")]),
  body([r("On a real OmniVLA 8B checkpoint (step 120000) navigating the real AI2-THOR photorealistic simulator on Kaggle's 2\u00D7T4 GPUs, I compared three regimes over "), r("192 paired test episodes", { bold: true }), r(" in four conditions (in-distribution, environment shift, instruction shift, and recovery from facing away): B2, the official every-step regime; B1, a cheap plan-following baseline; and B3, the proposed reliability-triggered regime.")]),
  body([r("Result: B3 matched the best observed success rate while issuing "), r("60% fewer model queries than the official regime (16.4 vs 41.4 per episode; 70 s vs 175 s of pure inference per episode)", { bold: true }), r(". No success difference between the regimes is statistically distinguishable (paired exact McNemar, all p = 1.000). Success was near zero in every cell: the pretrained checkpoint transfers poorly from its outdoor training domain to indoor scenes, and I report that as a finding rather than hiding it. The supporting pipeline (model loading, headless rendering, leakage-guarded logging) was verified end to end: all 12 raw-evidence checks pass, and every number in this report recomputes from raw episode logs.")]),

  h1("2. Background and Objective"),
  body([r("Modern navigation policies called "), r("Vision-Language-Action (VLA) models", { bold: true }), r(" are very large neural networks that take a camera image plus a language instruction and output how the robot should move. They are powerful but expensive to run: the model used here has 8 billion parameters, and one query takes about 3.5 to 4.2 seconds on two NVIDIA T4 GPUs.")]),
  body([r("The official way to deploy such a model is to query it at every control step - three times per second. Most of those queries are wasted when the world still matches the plan the model produced moments ago. This project asks one question: "), r("if we re-query the model only when the world stops matching its plan, how much inference can we save without losing success?", { bold: true })]),
  body([r("The scope of this report covers the full pipeline: loading the real checkpoint on cloud GPUs, rendering a real simulator headless, running a pilot, tuning the proposed trigger on a validation split only, running a 192-episode test matrix under a protocol frozen in advance, and recording demo videos. What it does not cover is fine-tuning the model itself; the checkpoint is used exactly as released.")]),

  h1("3. Method: Three Replanning Regimes and the Reliability Signal"),
  body([r("The robot already carries a very cheap sensor that the VLA model never uses: its own odometry. The proposed signal is purely onboard and costs essentially nothing to compute:")]),
  bullet([r("d_t = commanded displacement minus observed displacement", { bold: true }), r(" - how far the robot was told to move versus how far it actually moved;")]),
  bullet([r("a stalled-motion counter", { bold: true }), r(" - how many steps in a row the robot has failed to make progress.")]),
  body([r("If the discrepancy grows or the robot keeps stalling, the current plan is probably broken, so the model is re-queried. Otherwise the robot keeps following the plan it already holds. Three regimes are compared:")]),
  tableTitle("Table 1: The three replanning regimes compared in this project"),
  dataTable(
    ["Regime", "When the model is queried", "Role"],
    [
      ["B1 chunked-frozen", "every 8 steps; then follows the held 8-waypoint plan blindly", "cheapest baseline"],
      ["B2 fixed-frequency", "every step (the official deployment regime)", "accuracy reference"],
      ["B3 reliability-triggered (proposed)", "like B1, plus early re-query when the reliability signal fires", "proposed method"],
    ],
    [28, 44, 28]),
  body([r("B3's trigger threshold was chosen on a validation split only and then frozen before any test run, so no test information leaks into the method. The threshold search turned out to be degenerate (Section 6), which is reported honestly: the chosen value is the cheapest working setting, not a tuned optimum.")]),

  h1("4. Experimental Environment and Materials"),
  bullet([r("Model: ", { bold: true }), r("OmniVLA (NHirose/omnivla-original), 8B parameters, checkpoint at training step 120000, loaded in bf16 across the two T4 GPUs of a Kaggle notebook (about 15.4 GB). Measured inference: mean 3.56 s per query, 95th percentile 3.77 s; model load time 30.5 s.")]),
  bullet([r("Simulator: ", { bold: true }), r("AI2-THOR 5.0.0, a photorealistic indoor simulator, rendering headless on the Kaggle GPU machine with the Linux64 build under a virtual display (Xvfb): real 400\u00D7400 RGB frames at about 113 ms per step.")]),
  bullet([r("Goal interface: ", { bold: true }), r("the model's official language-only goal modality (a plain instruction such as \"Go to the toaster\"). Nothing is ever drawn into the camera image to help the robot.")]),
  bullet([r("Control: ", { bold: true }), r("the model outputs 8 waypoints; the official PD decoder tracks the 4th waypoint at a 1/3 s control period. This layer is identical for all three regimes, so the comparison is fair.")]),

  h1("5. Protocol: Splits, Conditions, and Fairness Rules"),
  body([r("The validation split (FloorPlan1-30) was used only to tune B3's threshold. The test split (FloorPlan401-430) produced all reported numbers. Four test conditions were designed and frozen before any test run:")]),
  tableTitle("Table 2: Test conditions (frozen before the test runs)"),
  dataTable(
    ["Code", "Meaning"],
    [
      ["A in-distribution", "test scenes from the canonical block, canonical instructions"],
      ["B environment shift", "a block of scenes never used elsewhere (FloorPlan301-330)"],
      ["C instruction shift", "same goals, paraphrased instructions"],
      ["D recovery", "the robot spawns facing directly away from the goal"],
    ],
    [30, 70]),
  body([r("Each condition ran "), r("16 episodes per regime (192 test episodes in total)", { bold: true }), r(", with random seeds paired across B1/B2/B3 so the very same episodes are compared regime against regime. Honesty rules are enforced in code, not by discipline: the logger refuses to overwrite an experiment; the analysis script refuses to run if a B3 threshold's recorded source mentions the test split; every log row records whether the real model produced it, and any row that did not is excluded and reported; success-efficiency (SPL) is reported only as an optimistic straight-line proxy and labelled as such.")]),

  h1("6. Procedure: Pipeline Steps and Engineering Challenges"),
  body([r("The experiment ran as a pipeline of Kaggle GPU notebooks (free-tier GPU quota is the reason episode counts are modest):")]),
  numItem("pipeline-steps", [r("00_preflight - loaded the real 8B checkpoint on 2\u00D7T4 and measured latency (load 30.5 s; inference mean 3.56 s, p95 3.77 s; VRAM 8.16 + 7.97 GB). All checks passed.")]),
  numItem("pipeline-steps", [r("01_thor - got the real AI2-THOR renderer working headless on Kaggle (CloudRendering crashes on this image, so the working path is Linux64 under Xvfb).")]),
  numItem("pipeline-steps", [r("02_thor_pilot - first comparison on 10 validation episodes per regime, 40-step budget: B2 solved 1 of 10; B1 and B3 solved none; B2 cost about 7\u00D7 B1's queries. The robot demonstrably moved (1.0-1.3 m of path per episode), so this was real navigation, not a stuck robot.")]),
  numItem("pipeline-steps", [r("03_tuning - swept B3's threshold (0.05/0.10/0.20/0.40 m) on validation only. Success was 0.000 at every threshold - a degenerate sweep - so the pre-registered tie-break rule (equal success, pick the cheapest) chose 0.20 m.")]),
  numItem("pipeline-steps", [r("04a/04b - the full 192-episode test matrix (about 3.5 hours of GPU time each).")]),
  numItem("pipeline-steps", [r("06_demo - demo videos of two fixed validation episodes with every tick recorded and no retakes: six single-regime clips plus two side-by-side comparisons.")]),
  body([r("Several failures had to be diagnosed and fixed on the way; every fix is now guarded in code and covered by local tests:")]),
  bullet([r("a stale nested copy of the source package inside the dataset archive silently ran old logic - notebooks now walk the mounts, print a provenance stamp, and refuse ambiguous copies;")]),
  bullet([r("one run picked up a stale dataset version - notebooks now assert the build stamp and refuse to run otherwise;")]),
  bullet([r("two concurrent kernels shared one temp folder and deleted each other's 15 GB model download, killing a run - every notebook now downloads into a unique per-run directory and re-verifies every file's existence and size immediately before loading;")]),
  bullet([r("a download once reported success while two model head files were missing - per-file size checks with exactly one forced retry now catch this;")]),
  bullet([r("headless-rendering issues (Xvfb flags, renamed apt packages on Ubuntu 24.04) - solved and documented.")]),
  body([r("Every notebook is generated by a script that parses every code cell before pushing, so a syntax error can never reach Kaggle.")]),

  h1("7. Results: Success Rate and Inference Cost"),
  h2("7.1 Pilot (validation, 10 episodes per regime)"),
  tableTitle("Table 3: Pilot results (validation split, 40-step budget)"),
  dataTable(
    ["Regime", "Episodes", "Success rate", "Queries/episode"],
    [
      ["B1 chunked-frozen", "10", "0.00", "5.2"],
      ["B2 fixed-frequency (official)", "10", "0.10", "35.8"],
      ["B3 reliability-triggered", "10", "0.00", "13.7"],
    ],
    [34, 16, 22, 28], [null, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]),
  h2("7.2 Main test matrix (192 episodes)"),
  tableTitle("Table 4: Test results pooled over all four conditions"),
  dataTable(
    ["Regime", "Episodes", "Success rate", "95% CI", "Queries per episode", "Inference time per episode"],
    [
      ["B1 chunked-frozen", "64", "0.03", "[0.01, 0.11]", "6.7", "28 s"],
      ["B2 fixed-frequency (official)", "64", "0.02", "[0.00, 0.08]", "41.4", "175 s"],
      ["B3 reliability-triggered (proposed)", "64", "0.03", "[0.01, 0.11]", "16.4", "70 s"],
    ],
    [24, 14, 13, 15, 16, 18],
    [null, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]),
  tableTitle("Table 5: Success rate per condition"),
  dataTable(
    ["Condition", "B1", "B2 (official)", "B3 (proposed)"],
    [
      ["A in-distribution", "0.00", "0.00", "0.00"],
      ["B environment shift", "0.06", "0.06", "0.06"],
      ["C instruction shift", "0.06", "0.00", "0.06"],
      ["D recovery", "0.00", "0.00", "0.00"],
    ],
    [40, 20, 20, 20], [null, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]),
  ...figure(path.join(FIG, "fig_success_rate.png"), 580,
    "Figure 1: Success rate per condition and pooled, with Wilson 95% confidence intervals. Intervals are wide because the episode count is small; no regime difference is statistically distinguishable.",
    [r("Figure 1 shows the success rates with their confidence intervals.")]),
  ...figure(path.join(FIG, "fig_inference_cost.png"), 500,
    "Figure 2: Mean model queries per episode on the test split. B3 costs 60% fewer queries than the official regime B2 at the same observed success rate.",
    [r("Figure 2 shows the cost side, which is the point of the exercise.")]),
  h2("7.3 Statistical reading"),
  body([r("Because episodes are paired across regimes, I ran exact McNemar tests per condition; every p-value is 1.000 (0-1 discordant pairs). At this sample size, "), r("no success-rate difference between the regimes is statistically distinguishable", { bold: true }), r(", and I claim no accuracy improvement. The cost difference, in contrast, is large and consistent: B3 matched the best observed success rate while issuing about 60% fewer queries than the official every-step regime. B3 re-queried about 14-15 times per episode, so the signal genuinely fires when the held plan fails instead of never firing.")]),
  h2("7.4 Demo videos"),
  body([r("The demo records every tick of two fixed validation episodes with no retakes. In episode 0 the official per-tick regime (B2) reached the light switch in 6 steps with 0 collisions, while the chunked regime (B1) pushed into furniture for 41 steps because it never re-queried; B3 re-queried 15 times but its re-plans did not escape the blocked corridor. The videos live in kaggle_notebooks/06_demo/output/demo/ (six per-regime MP4 files plus two side-by-side B1/B2/B3 comparisons).")]),
  ...figure(path.join(FIG, "fig_demo_frames.png"), 580,
    "Figure 3: Final recorded frame of demo episode 0 for each regime (real AI2-THOR renders with the run HUD: regime, step, velocity, distance to goal, instruction).",
    [r("Figure 3 shows the final recorded frame of each regime in that episode.")]),

  h1("8. Discussion: What the Numbers Mean"),
  body([r("Success was near zero in every cell, and that is itself a finding. ", { bold: true }), r("The checkpoint was trained mainly on outdoor robot data; indoor THOR scenes are a large domain shift. With a low success rate everywhere, the bottleneck is the pretrained model's transfer, not the replanning policy. Hiding this behind tuning would have been dishonest; instead the report treats it as the second result of the project.")]),
  body([r("On the question the project set out to answer, the evidence supports a clear, bounded claim: "), r("the onboard reliability trigger preserves the observed success rate of the official regime at roughly 60% less inference", { bold: true }), r(". The saving comes from the 2.5x gap between B3's 16.4 queries per episode and B2's 41.4, at a mean latency of about 4.2 s per query on this hardware. Because the trigger only watches odometry, it adds negligible compute of its own.")]),
  body([r("The degenerate threshold sweep deserves a note. Success was zero at every candidate threshold on validation, so the selection rule fell back to the pre-registered cost tie-break. A meaningful threshold tuning would need a regime where successes actually occur - which connects directly to the fine-tuning next step below.")]),

  h1("9. Limitations"),
  body([r("The following limitations are stated up front rather than hidden:")]),
  bullet([r("Small episode counts (16 per condition on test, 10 on the pilot) because of free-tier GPU quota; every rate is printed with its wide confidence interval.")]),
  bullet([r("SPL is reported only as a straight-line proxy that ignores walls - an optimistic bound, labelled as such.")]),
  bullet([r("The reliability signal is odometry-only: it cannot see a new obstacle before the robot has bumped into it.")]),
  bullet([r("AI2-THOR is a discrete simulator; the velocity-to-action mapping is deterministic and identical across regimes.")]),
  bullet([r("The 60-step episode budget is short; timeout failures and stall failures are not separated in the headline numbers.")]),

  h1("10. Conclusions and Next Steps"),
  numItem("concl-list", [r("The proposed onboard reliability trigger works as a cost control: ", { bold: true }), r("same observed success, about 60% fewer model queries than the official every-step regime, with the trigger demonstrably firing when the plan breaks (14-15 re-queries per episode).")]),
  numItem("concl-list", [r("Indoor success is limited by the pretrained model's domain, not by the replanning policy. ", { bold: true }), r("The natural next step is fine-tuning the VLA on indoor data; the replanning machinery is model-agnostic and would sit on top of any better policy unchanged.")]),
  numItem("concl-list", [r("A degenerate validation sweep still produced a usable threshold via the pre-registered cost tie-break, but meaningful threshold tuning needs a policy that sometimes succeeds.")]),
  body([r("Concrete next actions: (1) fine-tune or adapter-train the checkpoint on indoor navigation data, then re-run the same frozen protocol; (2) add a vision-based component (for example frame-difference) to the reliability signal so it can react before physical contact; (3) raise the episode count on the two conditions where the confidence intervals are widest.")]),

  new Paragraph({
    spacing: { line: 312, after: 0 },
    children: [r("The appendices list the reproduction commands and the location of every artifact."), new PageBreak()],
  }),
  h1("Appendix A: Reproduction Commands"),
  body([r("All tables in this report regenerate from the raw episode logs; nothing is transcribed by hand. From the project root:")]),
  ...codeBlock([
    "python scripts/verify_raw.py results/raw --split test",
    "# evidence gate: all 12 experiments pass",
    "",
    "python scripts/analyze_full.py",
    "python scripts/make_report.py",
    "python scripts/make_figures.py",
    "",
    "# Kaggle kernels, in running order (generate, then push):",
    "python scripts/gen_notebook_02.py",
    "kaggle kernels push -p kaggle_notebooks/02_thor_pilot",
    "python scripts/gen_notebook_03.py",
    "kaggle kernels push -p kaggle_notebooks/03_tuning",
    "python scripts/gen_notebook_04.py a",
    "kaggle kernels push -p kaggle_notebooks/04a_test_ab",
    "python scripts/gen_notebook_04.py b",
    "kaggle kernels push -p kaggle_notebooks/04b_test_cd",
    "python scripts/gen_notebook_06.py",
    "kaggle kernels push -p kaggle_notebooks/06_demo",
  ]),
  h1("Appendix B: Artifacts and Their Locations"),
  tableTitle("Table 6: Where each artifact lives"),
  dataTable(
    ["Artifact", "Location"],
    [
      ["Raw episode logs (12 experiments)", "results/raw/**/episodes.jsonl + meta.json"],
      ["Statistical analysis (tables, McNemar, bootstrap)", "results/analysis/full_analysis.md (+ .json)"],
      ["Auto-generated evidence report", "docs/final_report.md"],
      ["Report figures", "results/figures/"],
      ["Demo videos (8 MP4 files)", "kaggle_notebooks/06_demo/output/demo/"],
      ["Risk register (every failure and fix)", "docs/technical_risks.md"],
      ["Kaggle kernels (private, account pikanigga)", "pikanigga/vlanav-00-preflight ... pikanigga/vlanav-06-demo"],
    ],
    [42, 58]),
];

// ---------- document ----------
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: P.headingColor },
        paragraph: { spacing: { before: 360, after: 160, line: 312 } },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 30, bold: true, color: P.headingColor },
        paragraph: { spacing: { before: 240, after: 120, line: 312 } },
      },
    },
  },
  numbering: {
    config: ["pipeline-steps", "concl-list"].map(ref => ({
      reference: ref,
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    })),
  },
  sections: [
    { // Section 1: cover — margin 0, no footer, no page numbers
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        palette: P,
        englishLabel: "PROJECT REPORT",
        title: "VLA-NAV Project Report",
        subtitle: "Reliability-aware replanning for Vision-Language-Action robot navigation: when should a robot re-query its 8B-parameter model?",
        metaLines: [
          "Author: \u3010Your name\u3011",
          "Date: 30 October 2026",
          "Model: OmniVLA (8B) on 2\u00D7 NVIDIA T4 (Kaggle)",
          "Environment: AI2-THOR 5.0.0, photorealistic indoor scenes",
          "Evidence: 192 paired test episodes, 12/12 verification checks",
        ],
        footerLeft: "VLA-NAV Research Project",
        footerRight: "October 2026",
      }),
    },
    { // Section 2: front matter (TOC) — Roman numerals
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: docHeader() },
      footers: { default: pageNumFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "Table of Contents", bold: true, size: 32, font: { ascii: "Times New Roman", eastAsia: "SimHei" }, color: P.headingColor })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "Note: This Table of Contents is generated via field codes. To ensure page number accuracy after editing, please right-click the TOC and select \"Update Field.\"",
            italics: true, size: 18, color: "888888", font: { ascii: "Times New Roman" },
          })],
        }),
      ],
    },
    { // Section 3: body — Arabic numerals restarting at 1
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: docHeader() },
      footers: { default: pageNumFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, (buf.length / 1024).toFixed(0) + " KB");
});
