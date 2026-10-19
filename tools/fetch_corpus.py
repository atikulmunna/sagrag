#!/usr/bin/env python3
"""Fetch and format public-domain Stoic texts for ingestion.

Sources (public domain in the United States):
- Project Gutenberg #64576: Seneca, Minor Dialogues, tr. Aubrey Stewart (1889),
  written as one file per dialogue.
- Project Gutenberg #871: Epictetus, Golden Sayings, tr. Hastings Crossley
  (1909); the appended Hymn of Cleanthes gets its own file.
- Project Gutenberg #3794: Seneca, On Benefits, tr. Aubrey Stewart (1887).
- English Wikisource: Seneca, Moral Letters to Lucilius, tr. Richard Mott
  Gummere (1917 to 1925), letters 1 to 124 in one file.
- Project Gutenberg #45109: Epictetus, The Enchiridion, tr. Thomas Wentworth
  Higginson (introduction, bibliography and footnotes removed).
- Project Gutenberg #10661: Epictetus, A Selection from the Discourses, tr.
  George Long (his life of Epictetus and the appended Encheiridion removed).
- Project Gutenberg #2680: Marcus Aurelius, Meditations, tr. Meric Casaubon
  (introduction and the appendix of letters to Fronto removed).

Output format (what ingestion expects): UTF-8 .txt files in one domain folder,
one paragraph per line with a blank line between paragraphs (hard wraps
removed), headings kept as their own paragraphs, and publisher boilerplate,
translator prefaces, tables of contents, indexes, page markers, footnote
anchors and translator footnotes removed. File names start with the author,
because author-aware retrieval keys on them. A SOURCES.md manifest records
provenance (ingestion only reads .txt/.csv/.json, so it is not indexed).

Usage (from the repo root):
    python tools/fetch_corpus.py [--out data/docs/stoicism] [--cache DIR] [--only gutenberg|letters]
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = "SAG-RAG corpus fetcher (https://github.com/atikulmunna/sagrag)"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
LETTER_PAGE = "Moral letters to Lucilius/Letter {n}"
LETTER_COUNT = 124

# Section order of the Minor Dialogues volume, one entry per "BOOK" heading.
MINOR_DIALOGUES = [
    ("seneca_on_providence", "Of Providence"),
    ("seneca_on_the_firmness_of_the_wise", "On the Firmness of the Wise Man"),
    ("seneca_on_anger", "Of Anger, Book I"),
    ("seneca_on_anger", "Of Anger, Book II"),
    ("seneca_on_anger", "Of Anger, Book III"),
    ("seneca_consolation_to_marcia", "Of Consolation, to Marcia"),
    ("seneca_on_the_happy_life", "Of a Happy Life"),
    ("seneca_on_leisure", "Of Leisure"),
    ("seneca_on_peace_of_mind", "Of Peace of Mind"),
    ("seneca_on_the_shortness_of_life", "Of the Shortness of Life"),
    ("seneca_consolation_to_helvia", "Of Consolation, to Helvia"),
    ("seneca_consolation_to_polybius", "Of Consolation, to Polybius"),
    ("seneca_on_clemency", "Of Clemency, Book I"),
    ("seneca_on_clemency", "Of Clemency, Book II"),
]
_ORDINALS = "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH"
_BOOK_RE = re.compile(rf"^THE ({_ORDINALS}) BOOK OF THE DIALOGUES? OF L\. ANNAEUS SENECA", re.M)


# --- download ---------------------------------------------------------------


def http_get(url: str, cache: Path | None, name: str, retries: int = 3) -> str:
    """GET a URL as text, reusing a cached copy when one exists."""
    if cache is not None:
        cached = cache / name
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                (cache / name).write_text(text, encoding="utf-8")
            return text
        except OSError as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


# --- text cleanup -----------------------------------------------------------


def gutenberg_body(text: str) -> str:
    """Text between the Project Gutenberg START and END markers."""
    text = text.replace("\r\n", "\n")
    start = re.search(r"^\*\*\* ?START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    end = re.search(r"^\*\*\* ?END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    if not start or not end:
        raise ValueError("Project Gutenberg markers not found; the source layout changed")
    return text[start.end():end.start()]


def clean_inline(text: str) -> str:
    """Remove page markers, footnote anchors/blocks, and underscore italics."""
    text = re.sub(r"\[Footnote:.*?\]", "", text, flags=re.S)
    text = re.sub(r"\s*\[Greek:[^\]]*\]", "", text)
    text = re.sub(r"\{\d+\}", "", text)
    text = re.sub(r"\[Pg \d+\]", "", text)
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"(?<=\S)\[\d+\]", "", text)
    text = re.sub(r"_([^_\n]+(?:\n[^_\n]+)*)_", r"\1", text)
    return text


def paragraphs(text: str) -> list[str]:
    """Unwrap hard-wrapped text into paragraphs; drop translator footnotes."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        if re.match(r"^\s*\[\d+\]\s", block):
            continue  # a translator's footnote body, not the author's text
        para = re.sub(r"\s+", " ", clean_inline(block)).strip()
        if para:
            out.append(para)
    return out


def write_doc(path: Path, paras: list[str]) -> int:
    # Titles and translator credits live in SOURCES.md, not in the indexed
    # text: a header naming the author makes every file's first chunk match
    # any question that names the author.
    body = "\n\n".join(paras) + "\n"
    path.write_text(body, encoding="utf-8", newline="\n")
    return len(body)


# --- Project Gutenberg works -------------------------------------------------


def minor_dialogues(raw: str) -> dict[str, list[str]]:
    body = gutenberg_body(raw)
    end = body.rfind("THE END.")
    if end > 0:
        body = body[:end]  # drop the index
    starts = [m.start() for m in _BOOK_RE.finditer(body)]
    if len(starts) != len(MINOR_DIALOGUES):
        raise ValueError(f"expected {len(MINOR_DIALOGUES)} dialogue books, found {len(starts)}")
    files: dict[str, list[str]] = {}
    for (name, label), start, stop in zip(MINOR_DIALOGUES, starts, starts[1:] + [len(body)]):
        files.setdefault(name, [])
        files[name].append(label)
        # Drop the "THE NINTH BOOK OF THE DIALOGUES OF L. ANNAEUS SENECA,
        # ADDRESSED TO ..." heading for the same reason as the title line.
        files[name].extend(p for p in paragraphs(body[start:stop]) if not _BOOK_RE.match(p))
    return files


def golden_sayings(raw: str) -> dict[str, list[str]]:
    body = gutenberg_body(raw)
    contents_end = body.find("The Hymn of Cleanthes")
    first = re.compile(r"^I\s*$", re.M).search(body, contents_end)
    if contents_end < 0 or not first:
        raise ValueError("Golden Sayings layout changed")
    body = body[first.start():]
    hymn = body.rfind("(APPENDIX B)")
    sayings, hymn_text = body[:hymn], body[hymn:]
    hymn_paras = [p for p in paragraphs(hymn_text) if p not in ("(APPENDIX B)", "The Hymn of Cleanthes")]
    return {
        "epictetus_golden_sayings": paragraphs(sayings),
        "cleanthes_hymn_to_zeus": ["The Hymn of Cleanthes", *hymn_paras],
    }


def _between(body: str, start_re: str, end_re: str | None, what: str) -> str:
    """Slice ``body`` from the line matching ``start_re`` up to ``end_re``."""
    start = re.search(start_re, body, re.M)
    if not start:
        raise ValueError(f"{what}: start marker not found; the source layout changed")
    end = re.search(end_re, body[start.start():], re.M) if end_re else None
    return body[start.start(): start.start() + end.start()] if end else body[start.start():]


def enchiridion(raw: str) -> dict[str, list[str]]:
    # Drop the translator's introduction and bibliography (before the title)
    # and the footnotes section (after the text).
    body = _between(gutenberg_body(raw), r"^\s*THE ENCHIRIDION\s*$", r"^\s*Footnotes\s*$", "Enchiridion")
    return {"epictetus_enchiridion": paragraphs(body)}


def discourses(raw: str) -> dict[str, list[str]]:
    # Drop Long's life of Epictetus, and the appended Encheiridion (a second
    # translation of the separate Enchiridion file).
    body = _between(
        gutenberg_body(raw),
        r"^A SELECTION FROM THE DISCOURSES OF EPICTETUS\.\s*$",
        r"^THE ENCHEIRIDION, OR MANUAL\.\s*$",
        "Discourses",
    )
    # The volume heading names the author, which would make the first chunk
    # match every question about Epictetus; the body starts after it.
    paras = paragraphs(body)
    return {"epictetus_discourses": paras[1:] if paras and "DISCOURSES OF EPICTETUS" in paras[0] else paras}


def meditations(raw: str) -> dict[str, list[str]]:
    # Drop the editor's introduction and the appendix of letters to Fronto,
    # which are not the Meditations.
    body = _between(
        gutenberg_body(raw),
        # Start after the "MARCUS AURELIUS ANTONINUS THE ROMAN EMPEROR" title
        # line, for the same reason as the Discourses heading.
        r"^HIS FIRST BOOK\s*$",
        r"^APPENDIX\s*$",
        "Meditations",
    )
    return {"marcus_meditations": paragraphs(body)}


def on_benefits(raw: str) -> dict[str, list[str]]:
    body = gutenberg_body(raw)
    start = re.search(r"^L\. A\. SENECA\s*$", body, re.M)
    if not start:
        raise ValueError("On Benefits layout changed")
    return {"seneca_on_benefits": paragraphs(body[start.end():])}


# --- Wikisource letters ------------------------------------------------------


class _WikisourceText(HTMLParser):
    """Paragraph text of a Wikisource page, without navigation or notes."""

    SKIP_CLASSES = {
        "ws-noexport", "ws-header", "noprint", "reference", "references",
        "mw-references-wrap", "reflist", "pagenum", "ws-pagenum", "mw-editsection",
    }
    SKIP_TAGS = {"style", "script", "sup"}
    BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "tr", "blockquote", "br"}
    VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.parts: list[str] = []

    def _skipping(self) -> bool:
        return any(skip for _, skip in self.stack)

    def handle_starttag(self, tag, attrs):
        classes = set((dict(attrs).get("class") or "").split())
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")
        if tag in self.VOID_TAGS:
            return
        skip = tag in self.SKIP_TAGS or bool(classes & self.SKIP_CLASSES)
        self.stack.append((tag, skip))

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if not self._skipping():
            self.parts.append(data)


_SMALL_WORDS = frozenset("a an and as at but by for from in into nor of on or the to with".split())


def smart_title(text: str) -> str:
    """Title case that keeps short function words lower case."""
    words = text.lower().split()
    return " ".join(
        w if i and w in _SMALL_WORDS else w[:1].upper() + w[1:]
        for i, w in enumerate(words)
    )


def letter_paragraphs(page_html: str, n: int) -> list[str]:
    parser = _WikisourceText()
    parser.feed(page_html)
    text = html.unescape("".join(parser.parts))
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    if not paras:
        raise ValueError(f"letter {n}: no text extracted")
    heading = f"Letter {n}"
    # The title is the first short all-caps paragraph ("XIII. ON GROUNDLESS
    # FEARS"); volume headings such as "THE EPISTLES OF SENECA" may precede it.
    for i, para in enumerate(paras[:4]):
        letters = [c for c in para if c.isalpha()]
        if not letters or len(para) > 160 or sum(c.isupper() for c in letters) < 0.8 * len(letters):
            break
        if para.strip() == "THE EPISTLES OF SENECA":
            continue
        title = re.sub(r"^[IVXLC]+\.\s*", "", para).strip()
        heading = f"Letter {n}. {smart_title(title)}"
        paras = paras[i + 1:]
        break
    # Loeb section numbers ("1. I know that...") are layout, not text.
    paras = [re.sub(r"^\d+\.\s+", "", p) for p in paras]
    return [heading, *paras]


def moral_letters(cache: Path | None) -> dict[str, list[str]]:
    paras: list[str] = []
    for n in range(1, LETTER_COUNT + 1):
        query = urllib.parse.urlencode({
            "action": "parse", "page": LETTER_PAGE.format(n=n), "prop": "text",
            "format": "json", "formatversion": "2", "disableeditsection": "1",
        })
        name = f"letter{n:03d}.json"
        was_cached = cache is not None and (cache / name).exists()
        data = json.loads(http_get(f"{WIKISOURCE_API}?{query}", cache, name))
        if "error" in data:
            raise RuntimeError(f"letter {n}: {data['error'].get('info')}")
        paras.extend(letter_paragraphs(data["parse"]["text"], n))
        if not was_cached:
            time.sleep(0.3)  # be gentle with the Wikimedia API
    return {"seneca_moral_letters": paras}


# --- main --------------------------------------------------------------------

TITLES = {
    "seneca_on_providence": "Seneca, Of Providence (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_the_firmness_of_the_wise": "Seneca, On the Firmness of the Wise Man (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_anger": "Seneca, Of Anger, Books I to III (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_consolation_to_marcia": "Seneca, Of Consolation, to Marcia (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_the_happy_life": "Seneca, Of a Happy Life (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_leisure": "Seneca, Of Leisure (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_peace_of_mind": "Seneca, Of Peace of Mind (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_the_shortness_of_life": "Seneca, Of the Shortness of Life (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_consolation_to_helvia": "Seneca, Of Consolation, to Helvia (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_consolation_to_polybius": "Seneca, Of Consolation, to Polybius (Minor Dialogues), translated by Aubrey Stewart, 1889.",
    "seneca_on_clemency": "Seneca, Of Clemency, Books I and II, translated by Aubrey Stewart, 1889.",
    "epictetus_golden_sayings": "Epictetus, The Golden Sayings, translated by Hastings Crossley, 1909.",
    "cleanthes_hymn_to_zeus": "Cleanthes, Hymn to Zeus, translated by Hastings Crossley, 1909.",
    "seneca_on_benefits": "Seneca, On Benefits, translated by Aubrey Stewart, 1887.",
    "seneca_moral_letters": "Seneca, Moral Letters to Lucilius, letters 1 to 124, translated by Richard Mott Gummere, 1917 to 1925.",
    "epictetus_enchiridion": "Epictetus, The Enchiridion, translated by Thomas Wentworth Higginson, 1865.",
    "epictetus_discourses": "Epictetus, A Selection from the Discourses, translated by George Long, 1877.",
    "marcus_meditations": "Marcus Aurelius, Meditations, translated by Meric Casaubon, 1634.",
}
SOURCES = {
    "gutenberg-64576": "https://www.gutenberg.org/ebooks/64576",
    "gutenberg-871": "https://www.gutenberg.org/ebooks/871",
    "gutenberg-3794": "https://www.gutenberg.org/ebooks/3794",
    "wikisource-letters": "https://en.wikisource.org/wiki/Moral_letters_to_Lucilius",
    "gutenberg-45109": "https://www.gutenberg.org/ebooks/45109",
    "gutenberg-10661": "https://www.gutenberg.org/ebooks/10661",
    "gutenberg-2680": "https://www.gutenberg.org/ebooks/2680",
}
FILE_SOURCE = {name: "gutenberg-64576" for name, _ in MINOR_DIALOGUES}
FILE_SOURCE.update({
    "epictetus_golden_sayings": "gutenberg-871",
    "cleanthes_hymn_to_zeus": "gutenberg-871",
    "seneca_on_benefits": "gutenberg-3794",
    "seneca_moral_letters": "wikisource-letters",
    "epictetus_enchiridion": "gutenberg-45109",
    "epictetus_discourses": "gutenberg-10661",
    "marcus_meditations": "gutenberg-2680",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/docs/stoicism")
    parser.add_argument("--cache", default=None, help="directory to keep raw downloads for reruns")
    parser.add_argument("--only", choices=["gutenberg", "letters"], default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None

    docs: dict[str, list[str]] = {}
    if args.only in (None, "gutenberg"):
        docs.update(minor_dialogues(http_get(GUTENBERG_URL.format(id=64576), cache, "pg64576.txt")))
        docs.update(golden_sayings(http_get(GUTENBERG_URL.format(id=871), cache, "pg871.txt")))
        docs.update(on_benefits(http_get(GUTENBERG_URL.format(id=3794), cache, "pg3794.txt")))
        docs.update(enchiridion(http_get(GUTENBERG_URL.format(id=45109), cache, "pg45109.txt")))
        docs.update(discourses(http_get(GUTENBERG_URL.format(id=10661), cache, "pg10661.txt")))
        docs.update(meditations(http_get(GUTENBERG_URL.format(id=2680), cache, "pg2680.txt")))
    if args.only in (None, "letters"):
        docs.update(moral_letters(cache))

    rows = []
    for name, paras in docs.items():
        size = write_doc(out / f"{name}.txt", paras)
        words = sum(len(p.split()) for p in paras)
        rows.append((name, size, words, len(paras)))
        print(f"{name}.txt: {size:,} bytes, {words:,} words, {len(paras):,} paragraphs")

    manifest = ["# Corpus sources", "", "Written by `tools/fetch_corpus.py`. All texts are public domain in the United States.", ""]
    manifest += ["| File | Work | Source |", "|---|---|---|"]
    # List every fetched file present in the folder, not just this run's, so
    # a partial run (--only) keeps the manifest complete.
    for name in sorted(TITLES):
        if (out / f"{name}.txt").exists():
            manifest.append(f"| `{name}.txt` | {TITLES[name]} | {SOURCES[FILE_SOURCE[name]]} |")
    (out / "SOURCES.md").write_text("\n".join(manifest) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} files and SOURCES.md to {out}")


if __name__ == "__main__":
    sys.exit(main())
