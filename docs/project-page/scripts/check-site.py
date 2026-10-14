#!/usr/bin/env python3
"""Check local assets, anchors, HTML table values and optional HTTP serving.

No third-party dependencies. These checks do not replace browser visual QA.
"""
import argparse
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import struct
from urllib.parse import urljoin, urlsplit, unquote
from urllib.request import urlopen

SITE = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references, self.images, self.resources = [], set(), [], {}
        self.videos = []
        self.tables = {}
        self.table, self.row, self.cell = None, None, None
        self.title = []
        self.in_title = False
        self.bibtex = []
        self.in_bibtex = False

    def handle_starttag(self, tag, values):
        attrs = dict(values)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        for key in ("src", "href", "poster"):
            if attrs.get(key):
                self.references.add(attrs[key])
        if attrs.get("srcset"):
            self.references.update(item.strip().split()[0] for item in attrs["srcset"].split(","))
        if tag == "video":
            self.videos.append(attrs)
        if tag == "img":
            self.images.append(attrs)
        if "data-resource" in attrs:
            self.resources[attrs["data-resource"]] = (tag, attrs)
        if tag == "table":
            self.table = attrs.get("id", "unnamed")
            self.tables[self.table] = []
        if tag == "tr" and self.table:
            self.row = []
        if tag in ("th", "td") and self.row is not None:
            self.cell = []
        if tag == "h1":
            self.in_title = True
        if attrs.get("id") == "bibtex":
            self.in_bibtex = True

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)
        if self.in_title:
            self.title.append(value)
        if self.in_bibtex:
            self.bibtex.append(value)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.tables[self.table].append(self.row)
            self.row = None
        if tag == "table":
            self.table = None
        if tag == "h1":
            self.in_title = False
        if tag == "code":
            self.in_bibtex = False


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def latex_rows(path):
    return [row.split("&") for row in path.read_text().split("\\\\") if "&" in row]


def numeric(cell):
    match = re.search(r"\d+(?:\.\d+)?(?:B|M)?", cell)
    require(match is not None, f"No numeric value in {cell!r}")
    return match.group()


def compare_tables(page, source):
    names = [("π0.5", r"$\pi_{0.5}$"), ("GR00T-N1.7", "GR00T-N1.7"),
             ("openVLA-OFT", "openVLA-OFT"), ("VLA-Adapter", "VLA-Adapter"),
             ("ZR-0", "ZR-0"), ("VLA-Δ", r"VLA-$\Delta$")]
    rows = latex_rows(source / "tables" / "table1_libero.tex")
    web = page.tables["libero-table"][1:]
    require(len(web) == 6, "LIBERO must contain the selected six methods")
    for website, (name, token) in zip(web, names):
        original = next(row for row in rows if token in row[0])
        expected = [name] + [numeric(cell) for cell in original[1:]]
        require(website == expected, f"LIBERO mismatch: {website} != {expected}")
    names = [("X-VLA", "X-VLA"), ("π0", r"$\pi_{0}$"), ("π0.5", r"$\pi_{0.5}$"),
             ("Motus", "Motus"), ("ZR-0", "ZR-0"), ("VLA-Δ", r"VLA-$\Delta$")]
    rows = latex_rows(source / "tables" / "table2_robot_task_results.tex")
    web = page.tables["robotwin-table"][1:]
    require(len(web) == 6, "RoboTwin must contain all six methods")
    for website, (name, token) in zip(web, names):
        original = next(row for row in rows if token in row[0])
        expected = [name] + [numeric(cell) for cell in original[-2:]]
        require(website == expected, f"RoboTwin mismatch: {website} != {expected}")
    original = []
    for row in latex_rows(source / "tables" / "table4_ablation_libero.tex"):
        if len(row) != 7 or not any(name in row[0] for name in ("Dense VLM", "Difference Queries", "Query +", "VLA-")):
            continue
        flags = [re.search(r"\{(Yes|No)\}", cell).group(1) for cell in row[1:6]]
        original.append(flags + [numeric(row[-1])])
    web = page.tables["ablation-table"][1:]
    require(len(web) == len(original) == 8, "Ablation must contain eight variants")
    require([row[1:] for row in web] == original, "Ablation flags or results differ from manuscript")
    print("PASS: all 96 displayed benchmark/ablation values and flags match source tables")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional original manuscript directory")
    parser.add_argument("--site-root", type=Path, default=SITE,
                        help="Website directory to validate; defaults to the source website")
    parser.add_argument("--base-url", help="Server URL ending in /, including any project prefix")
    args = parser.parse_args()
    site = args.site_root.resolve()
    page = Page()
    html = (site / "index.html").read_text()
    page.feed(html)
    require(len(page.ids) == len(set(page.ids)), "Duplicate HTML ids")
    for reference in page.references:
        url = urlsplit(reference)
        if url.scheme or url.netloc:
            continue
        if not url.path:
            require(url.fragment in page.ids, f"Missing anchor {reference}")
            continue
        require(not url.path.startswith("/"), f"Root-relative asset breaks project paths: {reference}")
        require((site / unquote(url.path)).is_file(), f"Missing local resource {reference}")
    for image in page.images:
        require(bool(image.get("alt")), f"Missing alternative text: {image}")
        file = site / unquote(urlsplit(image["src"]).path)
        header = file.read_bytes()[:24]
        require(header[:8] == b"\x89PNG\r\n\x1a\n", f"Expected PNG fallback: {file}")
        dimensions = struct.unpack(">II", header[16:24])
        require(dimensions == (int(image["width"]), int(image["height"])), f"Incorrect image dimensions: {file}")
    for key in ("code", "arxiv"):
        tag, attrs = page.resources[key]
        require(tag == "span" and "href" not in attrs and attrs.get("aria-disabled") == "true", f"Unavailable {key} must not be a link")
    require("<iframe" not in html, "No unconfigured embedded player may be rendered")
    for video in page.videos:
        require(bool(video.get("src")), "Video must reference a real file")
        require("controls" in video and "playsinline" in video, "Video needs native, inline controls")
        require("autoplay" not in video, "Autoplay should start on viewport entry, not page load")
        if "data-hero-player" in video:
            require("muted" not in video and "loop" not in video, "Narrated hero video needs manual playback with its audio available")
        elif "data-motivation-video" in video:
            require("muted" in video, "Automatic video playback must start muted")
            require("loop" not in video, "The conceptual failure sequence should play once and hold its final state")
        else:
            require("muted" in video, "Automatic video playback must start muted")
            require("loop" in video, "Real-robot demonstrations need looping playback")
        require(bool(video.get("aria-label")), "Video needs an accessible title")
        require(int(video.get("width", 0)) > 0 and int(video.get("height", 0)) > 0, "Video needs dimensions")
    print(f"PASS: local resources, anchors, {len(page.images)} image descriptions/dimensions and unavailable resource states")
    if args.source:
        source = args.source.resolve()
        title = re.search(r"\\title\{(.+)\}", (source / "main.tex").read_text()).group(1)
        title = title.replace(r"$\Delta$", "Δ")
        require(" ".join("".join(page.title).split()) == title, "Title differs from main.tex")
        compare_tables(page, source)
        require(hashlib.sha256((site / "assets/paper/vla-delta.pdf").read_bytes()).digest() ==
                hashlib.sha256((source / "build/main.pdf").read_bytes()).digest(), "Packaged PDF changed")
        for tex in [source / "main.tex", *sorted((source / "sections").glob("*.tex"))]:
            for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text()):
                require((source / image).is_file(), f"Missing manuscript figure {image}")
        print("PASS: confirmed manuscript title, unchanged PDF and all manuscript figure paths")
    if args.base_url:
        base = args.base_url.rstrip("/") + "/"
        with urlopen(base, timeout=15) as response:
            require(response.status == 200, f"HTTP error at {base}")
            require(response.read().decode() == html, "Server returned the wrong index.html")
        count = 0
        for reference in sorted(page.references):
            url = urlsplit(reference)
            if url.scheme or url.netloc or not url.path:
                continue
            target = urljoin(base, url._replace(fragment="").geturl())
            require(target.startswith(base), f"Resource escapes project prefix: {target}")
            with urlopen(target, timeout=15) as response:
                require(response.status == 200, f"HTTP error: {target}")
                require(response.read() == (site / unquote(url.path)).read_bytes(), f"Wrong response content: {target}")
            count += 1
        print(f"PASS: index and {count} resource references served correctly under {base}")
    print("Browser layout, real clipboard and keyboard interaction require separate browser QA.")


if __name__ == "__main__":
    main()
