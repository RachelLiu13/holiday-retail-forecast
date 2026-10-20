"""Turn the PDF reports' content into HTML for upload to Google Docs.

Runs build_report.py / build_description.py, walks the reportlab story they build, and writes the
same content as HTML (Times New Roman 12pt). Images point at the public copies in the GitHub repo,
so Google Docs can fetch them during conversion.

Run:  python reports/build_gdocs_html.py   ->  reports/gdocs/*.html
"""
import html
import re
import runpy
from pathlib import Path

from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Table

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/RachelLiu13/holiday-retail-forecast/main/"
OUT = ROOT / "reports" / "gdocs"
OUT.mkdir(exist_ok=True)

FONT = "font-family:Times New Roman;"
CSS = {
    "p": FONT + "font-size:12pt;text-align:justify",
    "small": FONT + "font-size:10pt;color:#52514e",
    "h1": FONT + "font-size:16pt;color:#0d366b",
    "h2": FONT + "font-size:13pt;color:#0d366b",
    "title": FONT + "font-size:22pt;font-weight:bold;color:#0d366b",
    "subtitle": FONT + "font-size:13pt;color:#52514e",
    "li": FONT + "font-size:12pt",
    "td": FONT + "font-size:10pt",
    "th": FONT + "font-size:10pt;font-weight:bold;background:#f0efec",
    "pre": "font-family:Courier New;font-size:8pt",
}
# reportlab style name -> (html tag, css key)
ROLE = {"t": ("p", "title"), "st": ("p", "subtitle"), "sub": ("p", "subtitle"), "h1": ("h1", "h1"), "h2": ("h2", "h2"),
        "s": ("p", "small"), "small": ("p", "small"), "cap": ("p", "small")}


def markup(t: str) -> str:
    """reportlab paragraph markup -> HTML."""
    t = re.sub(r"<link href='([^']+)'[^>]*>", r"<a href='\1'>", t)
    t = t.replace("</link>", "</a>").replace("<super>", "<sup>").replace("</super>", "</sup>")
    return t


def cell_html(v, header):
    tag = "th" if header else "td"
    if isinstance(v, (list, tuple)):  # after layout a cell can hold a list of flowables
        text = "<br>".join(markup(x.text) for x in v if isinstance(x, Paragraph))
    else:
        text = markup(v.text) if isinstance(v, Paragraph) else html.escape(str(v))
    return f"<{tag} style=\"{CSS[tag]}\">{text}</{tag}>"


def image_html(img):
    path = Path(img.filename).resolve()
    url = RAW + path.relative_to(ROOT).as_posix()
    width = min(624, int(img.drawWidth * 96 / 72))  # Google Docs body is 6.5in = 624px wide
    height = int(width * img.drawHeight / img.drawWidth)
    return f"<p style=\"text-align:center\"><img src=\"{url}\" width=\"{width}\" height=\"{height}\"></p>"


def to_html(story, title):
    parts, in_list = [], False
    for f in story:
        is_bullet = isinstance(f, Paragraph) and f.bulletText
        if in_list and not is_bullet:
            parts.append("</ul>")
            in_list = False
        if isinstance(f, Preformatted):
            parts.append(f"<pre style=\"{CSS['pre']}\">{html.escape(chr(10).join(f.lines))}</pre>")
        elif isinstance(f, Paragraph):
            if is_bullet:
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append(f"<li style=\"{CSS['li']}\">{markup(f.text)}</li>")
                continue
            tag, key = ROLE.get(f.style.name, ("p", "p"))
            parts.append(f"<{tag} style=\"{CSS[key]}\">{markup(f.text)}</{tag}>")
        elif isinstance(f, Table):
            rows = f._cellvalues
            body = "".join("<tr>" + "".join(cell_html(v, i == 0) for v in r) + "</tr>" for i, r in enumerate(rows))
            parts.append(f"<table border=\"1\" style=\"border-collapse:collapse\">{body}</table>")
        elif isinstance(f, Image):
            parts.append(image_html(f))
        elif isinstance(f, PageBreak):
            parts.append("<p style=\"page-break-before:always;\"></p>")
    if in_list:
        parts.append("</ul>")
    return (f"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title></head>"
            f"<body style=\"{FONT}font-size:12pt;\">" + "\n".join(parts) + "</body></html>")


# doc.build() consumes the story list, so keep a copy of what each script passes to it
_captured = {}
_build = SimpleDocTemplate.build


def _capture(self, flowables, *a, **k):
    _captured["story"] = list(flowables)
    return _build(self, flowables, *a, **k)


SimpleDocTemplate.build = _capture

for script, name, title in [("build_report.py", "Written_Report.html", "Forecasting the 2026 U.S. Holiday Retail Season - Written Report"),
                            ("build_description.py", "Project_Description.html", "Holiday Retail Planner 2026 - Project Description")]:
    runpy.run_path(str(ROOT / "reports" / script), run_name="__build__")
    (OUT / name).write_text(to_html(_captured["story"], title), encoding="utf-8")
    print(OUT / name, len((OUT / name).read_text()))
