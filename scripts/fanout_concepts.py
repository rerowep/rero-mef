#!/usr/bin/env python
# SPDX-FileCopyrightText: Fondation RERO+
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the fan-out concept dataset from the two public APIs.

A BNF fan-out group is one BNF number claimed by several GND concepts. The IdRef
concept holding that number then faces a choice the rules refuse to make, so it
stays unlinked however obvious the pairing looks.

Everything here comes from mef.rero.ch and bib.rero.ch, so the dataset can be
rebuilt at any time without a database:

    uv run python scripts/fanout_concepts.py all -o data/fanout.json

The work is staged and every stage caches its raw result next to the output, so
a rerun resumes instead of starting over:

    topics   bib.rero.ch  remote_entities type:bf:Topic         ~345 requests
    gnd      mef.rero.ch  concepts/gnd carrying a BNF number    ~466 requests
    counts   bib.rero.ch  documents per fan-out IdRef pid      ~3400 requests
    build    join the three and emit the dataset                       local

Counting from the GND side is what keeps this affordable: asking mef for every
IdRef number one at a time would be ~34000 requests, enumerating the GND records
that carry a BNF number at all is 466.

Two columns of the original page are NOT recomputable here and are deliberately
absent: the `075 $b saf` linkage, which needs a GND transformation that is not
deployed, and the cascade suggestions, which come from a separate heuristic.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

MEF = "https://mef.rero.ch/api"
BIB = "https://bib.rero.ch/api"
PAGE = 100
#: Largest partition fetched in one request. Well under any result window, and
#: the point is to never page at all.
PARTITION = 2000
#: Modest concurrency: these are someone's production services.
WORKERS = 6

#: `FRBNF` plus eight digits, or the same number as a BNF ark uri. Only the eight
#: digits are stable: IdRef writes the number bare, GND sometimes keeps the
#: trailing check character, and both also store it as an ark.
BNF_NUMBER = re.compile(r"^FRBNF(\d{8})|^https?://[\w.]+\.bnf\.fr/ark:/12148/cb(\d{8})")
MATCH_TYPES = ("exactMatch", "closeMatch")


def bnf_number(value):
    """Reduce an identifier value to `FRBNF` plus its eight digits."""
    if match := BNF_NUMBER.match(value or ""):
        return f"FRBNF{match[1] or match[2]}"
    return None


def numbers_of(identified_by):
    """Get the BNF numbers an `identifiedBy` list states as `bf:Nbn`."""
    return {
        number
        for entry in identified_by or []
        if entry.get("type") == "bf:Nbn"
        if (number := bnf_number(entry.get("value")))
    }


def get(url, tries=4):
    """Fetch one JSON document, retrying on a transient failure."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == tries - 1:
                raise SystemExit(f"giving up on {url}: {error}") from error
            time.sleep(2 * (attempt + 1))
    return None


def total_of(payload):
    """Read `hits.total` in either of the two shapes the API returns."""
    total = payload["hits"]["total"]
    return total["value"] if isinstance(total, dict) else total


def scan(base, endpoint, query, fields=None):
    """Yield every record of a query once, one page at a time.

    Paging deep into a query the search engine has not ordered stably returns
    some records twice and misses others: the mef GND scan came back with 10298
    duplicate rows out of 46479. Records are therefore yielded only the first
    time their pid is seen, and the count of what was skipped is reported so a
    scan that loses records does not pass unnoticed.

    :param base: API root.
    :param endpoint: endpoint path, e.g. `concepts/gnd`.
    :param query: value of the `q` parameter.
    :param fields: progress label.
    """
    params = urllib.parse.urlencode({"q": query, "size": PAGE})
    first = get(f"{base}/{endpoint}/?{params}")
    total = total_of(first)
    pages = -(-total // PAGE)
    click.secho(f"  {fields or endpoint}: {total} records, {pages} pages", err=True)
    seen = set()

    def fresh(payload):
        """Yield the records of one page whose pid has not been seen yet."""
        for hit in payload["hits"]["hits"]:
            record = hit["metadata"]
            if record["pid"] not in seen:
                seen.add(record["pid"])
                yield record

    yield from fresh(first)

    def page(number):
        query_string = urllib.parse.urlencode({"q": query, "size": PAGE, "page": number})
        return get(f"{base}/{endpoint}/?{query_string}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, payload in enumerate(pool.map(page, range(2, pages + 1)), start=2):
            if done % 50 == 0:
                click.secho(f"    page {done}/{pages}", err=True)
            yield from fresh(payload)
    if len(seen) != total:
        click.secho(f"    {total - len(seen)} of {total} records never came back", fg="yellow", err=True)


def stage_topics(cache):
    """Every concept bib.rero.ch actually uses, with its BNF number."""
    topics = []
    for entity in scan(BIB, "remote_entities", 'type:"bf:Topic"', "bib.rero.ch bf:Topic"):
        idref = entity.get("idref")
        if not idref:
            continue
        topics.append(
            {
                "pid": idref["pid"],
                "heading": idref.get("authorized_access_point"),
                "numbers": sorted(numbers_of(idref.get("identifiedBy"))),
                # the GND already clustered with it in production, if any
                "gnd": (entity.get("gnd") or {}).get("pid"),
                "gnd_heading": (entity.get("gnd") or {}).get("authorized_access_point"),
            }
        )
    cache.write_text(json.dumps(topics, ensure_ascii=False))
    click.secho(f"  -> {len(topics)} topics", err=True)
    return topics


def scan_gnd_by_prefix(prefix="", depth=0):
    """Yield every GND concept whose BNF number starts with a prefix.

    `concepts/gnd` has no sort option, so one query paged 466 pages deep both
    repeats and loses records: 9953 of 46501 never came back. Partitioning by
    the digits of the number instead keeps every partition small enough to
    arrive in a single request, which is complete by construction and costs
    fewer requests than the paging did.

    A record naming numbers with different prefixes turns up in more than one
    partition; the caller keys on the pid, so that is harmless.

    :param prefix: digits already fixed after `FRBNF`.
    :param depth: recursion depth, for the progress line.
    """
    pattern = f"FRBNF{prefix}*"
    query = f"exactMatch.identifiedBy.value:{pattern} OR closeMatch.identifiedBy.value:{pattern}"
    params = urllib.parse.urlencode({"q": query, "size": 1})
    total = total_of(get(f"{MEF}/concepts/gnd/?{params}"))
    if not total:
        return
    if total <= PARTITION:
        if depth <= 2:
            click.secho(f"    {pattern}: {total}", err=True)
        params = urllib.parse.urlencode({"q": query, "size": total})
        payload = get(f"{MEF}/concepts/gnd/?{params}")
        got = payload["hits"]["hits"]
        if len(got) != total:
            click.secho(f"    {pattern}: got {len(got)} of {total}", fg="yellow", err=True)
        yield from (hit["metadata"] for hit in got)
        return
    if len(prefix) >= 8:
        click.secho(f"    {pattern}: {total} cannot be split further", fg="red", err=True)
        return
    for digit in "0123456789":
        yield from scan_gnd_by_prefix(prefix + digit, depth + 1)


def stage_gnd(cache):
    """Every GND concept carrying a BNF number, with the number it decides on."""
    records = []
    seen = set()
    click.secho("  mef.rero.ch GND with BNF, by number prefix:", err=True)
    for concept in scan_gnd_by_prefix():
        if concept["pid"] in seen:
            continue
        seen.add(concept["pid"])
        for match_type in MATCH_TYPES:
            blocks = concept.get(match_type) or []
            numbers = set()
            for block in blocks:
                numbers |= numbers_of(block.get("identifiedBy"))
            if not numbers:
                continue
            # `exactMatch` wins over `closeMatch`; several distinct numbers in the
            # winning block type state no usable identifier at all.
            records.append(
                {
                    "pid": concept["pid"],
                    "heading": concept.get("authorized_access_point"),
                    "level": match_type,
                    "numbers": sorted(numbers) if len(numbers) == 1 else [],
                    "ambiguous": len(numbers) > 1,
                    "all_numbers": sorted(numbers),
                }
            )
            break
    cache.write_text(json.dumps(records, ensure_ascii=False))
    click.secho(f"  -> {len(records)} GND records", err=True)
    return records


def fan_out(topics, gnd_records):
    """Join the two sides: the concepts whose number several GND records claim."""
    by_number = {}
    for record in gnd_records:
        for number in record["all_numbers"]:
            by_number.setdefault(number, []).append(record)

    rows = []
    for topic in topics:
        candidates = {}
        for number in topic["numbers"]:
            for record in by_number.get(number, []):
                candidates[record["pid"]] = record
        if len(candidates) < 2:
            continue
        rows.append({**topic, "candidates": list(candidates.values())})
    return rows


def stage_counts(rows, cache):
    """The number of bib.rero.ch documents each concept is a subject of."""
    existing = json.loads(cache.read_text()) if cache.exists() else {}

    def count(pid):
        params = urllib.parse.urlencode({"q": f"subjects.entity.pids.idref:{pid}", "size": 1})
        return pid, total_of(get(f"{BIB}/documents/?{params}"))

    todo = [row["pid"] for row in rows if row["pid"] not in existing]
    click.secho(f"  {len(todo)} counts to fetch ({len(existing)} cached)", err=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, (pid, total) in enumerate(pool.map(count, todo), start=1):
            existing[pid] = total
            if done % 200 == 0:
                click.secho(f"    {done}/{len(todo)}", err=True)
                cache.write_text(json.dumps(existing))
    cache.write_text(json.dumps(existing))
    return existing


def stage_idref(gnd_records, topics, cache):
    """Resolve the BNF numbers of the inverse fan-out to their IdRef concepts.

    The forward direction only ever needs concepts bib.rero.ch uses, so the
    topics cache answers most numbers for free. The rest belong to IdRef
    concepts the catalogue does not use, and have to be asked for.

    :returns: BNF number -> {pid, heading}.
    """
    known = {}
    for topic in topics:
        for number in topic["numbers"]:
            known.setdefault(number, {"pid": topic["pid"], "heading": topic["heading"]})

    resolved = json.loads(cache.read_text()) if cache.exists() else {}
    wanted = {number for record in gnd_records if len(record["all_numbers"]) > 1 for number in record["all_numbers"]}
    todo = sorted(number for number in wanted if number not in known and number not in resolved)
    click.secho(
        f"  {len(wanted)} numbers involved, {len(wanted) - len(todo) - len(resolved)} from the topics cache, "
        f"{len(todo)} to look up",
        err=True,
    )

    def lookup(number):
        """Get the single IdRef concept stating one BNF number."""
        params = urllib.parse.urlencode({"q": f'identifiedBy.value:"{number}"', "size": 2})
        hits = get(f"{MEF}/concepts/idref/?{params}")["hits"]["hits"]
        if len(hits) != 1:
            return number, None
        found = hits[0]["metadata"]
        return number, {"pid": found["pid"], "heading": found.get("authorized_access_point")}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, (number, found) in enumerate(pool.map(lookup, todo), start=1):
            resolved[number] = found
            if done % 200 == 0:
                click.secho(f"    {done}/{len(todo)}", err=True)
                cache.write_text(json.dumps(resolved, ensure_ascii=False))
    cache.write_text(json.dumps(resolved, ensure_ascii=False))
    return known | {number: found for number, found in resolved.items() if found}


def inverse(gnd_records, idref_of, counts):
    """The mirror direction: one GND record, several IdRef concepts.

    A GND record naming two different BNF numbers states no usable identifier
    at all, so it links to none of them, however obvious a pairing looks.
    """
    out = []
    for record in gnd_records:
        if len(record["all_numbers"]) < 2:
            continue
        sides = [
            {"p": side["pid"], "h": side["heading"], "b": number, "n": counts.get(side["pid"], 0)}
            for number in record["all_numbers"]
            if (side := idref_of.get(number))
        ]
        # two numbers that resolve to the same concept are one claim, not a fan-out
        if len({side["p"] for side in sides}) < 2:
            continue
        out.append(
            {
                "g": record["pid"],
                "gh": record["heading"],
                "lvl": record["level"],
                "t": sorted(sides, key=lambda side: -side["n"]),
                "n": sum(side["n"] for side in sides),
            }
        )
    out.sort(key=lambda row: (-row["n"], row["gh"] or ""))
    return out


def build(rows, counts):
    """Shape the dataset the page renders."""
    out = []
    for row in rows:
        candidates = sorted(row["candidates"], key=lambda c: c["heading"] or "")
        chosen = row["gnd"]
        out.append(
            {
                "n": counts.get(row["pid"], 0),
                "h": row["heading"],
                "p": row["pid"],
                # `already`: production already clusters it with a GND record.
                "s": "already" if chosen else "open",
                "g": f"{chosen} {row['gnd_heading']}" if chosen else None,
                "c": [f"{c['pid']} {c['heading']}" for c in candidates if c["pid"] != chosen],
                "t": len(candidates),
            }
        )
    out.sort(key=lambda row: -row["n"])
    return out


def main():
    """Run the requested stages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["topics", "gnd", "idref", "counts", "build", "all"])
    parser.add_argument("-o", "--out", default="fanout.json", help="dataset to write")
    args = parser.parse_args()

    out = Path(args.out)
    cache = out.parent
    topics_cache, gnd_cache, counts_cache, idref_cache = (
        cache / "fanout_topics.json",
        cache / "fanout_gnd.json",
        cache / "fanout_counts.json",
        cache / "fanout_idref.json",
    )
    run = {"all": {"topics", "gnd", "idref", "counts", "build"}}.get(args.stage, {args.stage})

    # A stage only reads the caches the stages after it need, so each one can be
    # run on its own.
    if "topics" in run:
        click.secho("topics:", err=True)
        stage_topics(topics_cache)
    if "gnd" in run:
        click.secho("gnd:", err=True)
        stage_gnd(gnd_cache)
    if not run & {"idref", "counts", "build"}:
        return

    topics = json.loads(topics_cache.read_text())
    gnd_records = json.loads(gnd_cache.read_text())
    rows = fan_out(topics, gnd_records)
    click.secho(f"fan-out concepts used by bib.rero.ch: {len(rows)}", err=True)

    # Cache-aware: with a complete cache it asks for nothing, so it needs no stage
    # guard of its own.
    click.secho("idref:", err=True)
    idref_of = stage_idref(gnd_records, topics, idref_cache)

    # The inverse rows carry their own IdRef concepts, which the forward set does
    # not always contain, so they are counted too.
    mirror_pids = [{"pid": side["pid"]} for side in idref_of.values()]
    click.secho("counts:", err=True)
    counts = stage_counts(rows + mirror_pids, counts_cache) if "counts" in run else json.loads(counts_cache.read_text())

    if "build" in run:
        dataset = build(rows, counts)
        mirror = inverse(gnd_records, idref_of, counts)
        payload = {
            "generated": time.strftime("%Y-%m-%d"),
            "rows": dataset,
            "documents": sum(row["n"] for row in dataset),
            "already": sum(row["s"] == "already" for row in dataset),
            "mirror": mirror,
            "mirror_documents": sum(row["n"] for row in mirror),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        click.secho(
            f"wrote {out}: {len(dataset)} concepts, {payload['documents']} documents, "
            f"{payload['already']} already linked; {len(mirror)} inverse groups, "
            f"{payload['mirror_documents']} documents",
            fg="green",
            err=True,
        )


if __name__ == "__main__":
    main()
