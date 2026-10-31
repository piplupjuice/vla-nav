"""Post-process project_report.docx page numbering (WPS compatibility).

1. remove empty <w:pgNumType/> (docx-js emits it in sections without pageNumbers)
2. patch footer PAGE fields with explicit format switches, mapped per section:
   upperRoman section -> PAGE \\* ROMAN \\* MERGEFORMAT
   decimal section    -> PAGE \\* arabic \\* MERGEFORMAT
"""
import re
import shutil
import sys
import zipfile

DOCX = sys.argv[1] if len(sys.argv) > 1 else "docs/project_report.docx"
TMP = DOCX + ".tmp"

with zipfile.ZipFile(DOCX) as z:
    names = z.namelist()
    data = {n: z.read(n) for n in names}

doc = data["word/document.xml"].decode("utf-8")
doc = doc.replace("<w:pgNumType/>", "")

# remove the stray first TOC entry (hyperlink with a dirty, number-less PAGEREF)
# that add_toc_placeholders.py leaves inside the TOC field paragraph
sep = doc.find('w:fldCharType="separate"')
if sep != -1:
    h_start = doc.find("<w:hyperlink", sep)
    h_end = doc.find("</w:hyperlink>", sep)
    if h_start != -1 and h_end != -1:
        block = doc[h_start:h_end]
        if 'w:anchor="_Toc' in block and 'w:dirty="true"' in block:
            doc = doc[:h_start] + doc[h_end + len("</w:hyperlink>"):]
            print("removed stray TOC hyperlink entry")

rels = data["word/_rels/document.xml.rels"].decode("utf-8")
rid_to_target = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))

# map each sectPr's page-number format to its footer part
patch_plan = {}  # footer file -> switch
for sect in re.findall(r"<w:sectPr[^>]*>.*?</w:sectPr>", doc, re.S):
    fmt_m = re.search(r'<w:pgNumType[^>]*w:fmt="([^"]+)"', sect)
    if not fmt_m:
        continue
    switch = {"upperRoman": "ROMAN", "decimal": "arabic"}.get(fmt_m.group(1))
    if not switch:
        continue
    for rid in re.findall(r'<w:footerReference[^>]*r:id="([^"]+)"', sect):
        target = rid_to_target.get(rid, "")
        if target:
            patch_plan.setdefault("word/" + target.lstrip("/"), switch)

patched = []
for fname, switch in patch_plan.items():
    if fname not in data:
        continue
    xml = data[fname].decode("utf-8")
    new_xml, n = re.subn(
        r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
        rf"\1 PAGE \\* {switch} \\* MERGEFORMAT \2",
        xml)
    if n:
        data[fname] = new_xml.encode("utf-8")
        patched.append(f"{fname} -> {switch} ({n} field)")

data["word/document.xml"] = doc.encode("utf-8")

with zipfile.ZipFile(TMP, "w", zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, data[n])
shutil.move(TMP, DOCX)
print("removed empty pgNumType; patched footers:")
for p in patched:
    print("  -", p)
if not patched:
    print("  WARNING: no footer fields patched")
