#!/usr/bin/env python3
"""
Fetch common (vernacular) names for Latin species names using Wikidata and Wikipedia.

Creates/updates three artifacts:
 - common_names_cache.json : cached mapping {latin: {wikidata_id, wikipedia_title, common_names, sources}}
 - common_names_mapping.csv : CSV with latin_name and semicolon-separated common names
 - enhanced_with_common_names_prompts.json : same shape as enhanced_mushroom_prompts.json but with added common-name prompts

Usage:
    python fetch_common_names.py --test 10
    python fetch_common_names.py             # full run (respects cache)

Notes:
 - Respects cache and writes results incrementally.
 - Uses Wikidata API to find aliases and English Wikipedia lead to extract additional common-name phrases.
 - Be polite: sleeps between requests. If you have many species, consider running on a machine with a persistent IP and respecting rate limits.
"""

import argparse
import json
import time
import csv
import re
import os
from urllib.parse import quote

import requests

ROOT = os.path.dirname(__file__)
ENHANCED_PROMPTS = os.path.join(ROOT, 'enhanced_mushroom_prompts.json')
OUT_PROMPTS = os.path.join(ROOT, 'enhanced_with_common_names_prompts.json')
CACHE_JSON = os.path.join(ROOT, 'common_names_cache.json')
MAPPING_CSV = os.path.join(ROOT, 'common_names_mapping.csv')
LOG_FILE = os.path.join(ROOT, 'fetch_common_names.log')

HEADERS = {'User-Agent': 'mushroom-prompts-bot/1.0 (mailto:you@example.com)'}

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}\n"
    print(line, end='')
    with open(LOG_FILE, 'a') as f:
        f.write(line)

def load_json(path):
    if os.path.exists(path):
        with open(path,'r') as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path,'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fetch_wikidata_entity(latin):
    """Search Wikidata for the latin label and return the top entity id and basic data."""
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action':'wbsearchentities',
        'format':'json',
        'language':'en',
        'search': latin,
        'limit': 1,
        'type':'item'
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get('search'):
        return data['search'][0]
    return None

def get_wikidata_aliases(qid):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action':'wbgetentities',
        'format':'json',
        'ids': qid,
        'props':'aliases|sitelinks|labels',
        'languages':'en'
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    ent = data.get('entities', {}).get(qid, {})
    aliases = []
    if 'aliases' in ent and 'en' in ent['aliases']:
        aliases = [a['value'] for a in ent['aliases']['en']]
    # label may sometimes be a vernacular
    if 'labels' in ent and 'en' in ent['labels']:
        label = ent['labels']['en']['value']
        if label:
            aliases.append(label)
    sitelinks = ent.get('sitelinks', {})
    wiki_title = None
    if 'enwiki' in sitelinks:
        wiki_title = sitelinks['enwiki']['title']
    return list(dict.fromkeys([a.strip() for a in aliases if a.strip()])), wiki_title

def fetch_wikipedia_common_phrases(title):
    """Fetch Wikipedia summary and try to extract 'commonly known as' or parenthetical common names."""
    if not title:
        return []
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        extract = data.get('extract', '')
        # patterns: "commonly known as the X" or parentheses immediately after latin name
        commons = set()
        # parentheses right after scientific name: e.g. "Amanita muscaria, commonly known as the fly agaric"
        m = re.search(r'commonly known as ([^.,;(\n]+)', extract, flags=re.IGNORECASE)
        if m:
            commons.add(m.group(1).strip())
        # parenthetical common names: e.g. "(commonly known as the fly agaric)" or "(fly agaric)"
        for pm in re.findall(r'\(([^)]+)\)', extract):
            if len(pm) < 80 and not re.search(r'\d{4}', pm):
                # heuristics: exclude long parentheses or dates
                commons.add(pm.strip())
        # Also try phrases like "also called X"
        for m2 in re.findall(r'also (?:called|known as) ([^.,;\n]+)', extract, flags=re.IGNORECASE):
            commons.add(m2.strip())
        # Clean common names: split on ' or ' or ',' when multiple
        results = []
        for c in commons:
            parts = re.split(r',| or |;|\/| and ', c)
            for p in parts:
                p = p.strip(' \t\n"\'')
                if p and len(p) > 1 and len(p) < 80:
                    results.append(p)
        return list(dict.fromkeys(results))
    except Exception:
        return []

def _is_mostly_latin(s: str) -> bool:
    # require at least one ASCII/Latin letter and no strong presence of non-Latin scripts
    if not re.search(r'[A-Za-z]', s):
        return False
    # quick reject if contains CJK or Arabic or Cyrillic characters
    if re.search(r'[\u4E00-\u9FFF\u0600-\u06FF\u0400-\u04FF]', s):
        return False
    return True


def find_common_names_for(latin, english_only=False):
    # Try wikidata search
    try:
        ent = fetch_wikidata_entity(latin)
    except Exception as e:
        log(f"Wikidata search failed for {latin}: {e}")
        ent = None
    time.sleep(0.3)
    qid = None
    aliases = []
    wiki_title = None
    wiki_phrases = []
    if ent:
        qid = ent.get('id')
        try:
            aliases, wiki_title = get_wikidata_aliases(qid)
        except Exception as e:
            log(f"Wikidata get failed for {latin} ({qid}): {e}")
        time.sleep(0.3)
    if wiki_title:
        try:
            wiki_phrases = fetch_wikipedia_common_phrases(wiki_title)
        except Exception as e:
            log(f"Wikipedia fetch failed for {wiki_title}: {e}")
        time.sleep(0.3)

    # Combine aliases and wiki_phrases, prefer shorter phrases, dedupe
    candidates = []
    for a in aliases + wiki_phrases:
        a = a.strip()
        if not a:
            continue
        if a.lower() == latin.lower():
            continue
        # skip if clearly scientific (contains binomial)
        if re.search(r'\b[A-Z][a-z]+\s[a-z]+', a):
            # looks like a Latin name; skip
            continue
        # remove trailing punctuation
        a = a.strip(' .;')
        if 1 < len(a) < 80:
            candidates.append(a)
    # simple cleanup and keep unique
    seen = set()
    out = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        # If english_only, filter to Latin-script/English-looking names
        if english_only and not _is_mostly_latin(c):
            continue
        seen.add(key)
        out.append(c)
    return {'wikidata_id': qid, 'wikipedia_title': wiki_title, 'common_names': out, 'sources': {'aliases': len(aliases)>0, 'wiki_phrases': len(wiki_phrases)>0}}

def append_common_name_prompts(prompts_list, common_name):
    # heuristics for a few common-name prompt templates
    templates = [
        f"a photo of {common_name}",
        f"a photograph of {common_name}",
        f"{common_name} mushroom",
        f"{common_name} fungus",
        f"photo of {common_name} mushroom",
        f"field guide photo of {common_name}",
    ]
    # avoid duplicates
    existing = set(p.lower() for p in prompts_list)
    for t in templates:
        if t.lower() not in existing:
            prompts_list.append(t)
            existing.add(t.lower())

def main(args):
    enhanced = load_json(ENHANCED_PROMPTS)
    cache = load_json(CACHE_JSON)

    species = sorted(enhanced.keys())
    if args.test:
        species = species[:args.test]

    updated = 0
    for i, s in enumerate(species, start=1):
        if s in cache and cache[s].get('common_names'):
            log(f"[{i}/{len(species)}] {s} -> cached ({len(cache[s]['common_names'])} names)")
            continue
        log(f"[{i}/{len(species)}] Fetching common names for {s}...")
        try:
            info = find_common_names_for(s)
        except Exception as e:
            log(f"Error fetching for {s}: {e}")
            info = {'wikidata_id': None, 'wikipedia_title': None, 'common_names': [], 'sources': {}}
        cache[s] = info
        # write cache incrementally
        save_json(CACHE_JSON, cache)
        # also update CSV mapping row
        with open(MAPPING_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['latin_name','wikidata_id','wikipedia_title','common_names'])
            for k in sorted(cache.keys()):
                row = [k, cache[k].get('wikidata_id') or '', cache[k].get('wikipedia_title') or '', ';'.join(cache[k].get('common_names', []))]
                writer.writerow(row)
        if info.get('common_names'):
            updated += 1
            log(f" -> found {len(info['common_names'])} names: {info['common_names']}")
        else:
            log(" -> none found")
        # be polite
        time.sleep(0.4)

    # Build new prompts JSON by appending common-name prompts where found
    out_prompts = {}
    for s, prompts in enhanced.items():
        p = list(prompts)
        cn_info = cache.get(s, {})
        for cn in cn_info.get('common_names', []):
            append_common_name_prompts(p, cn)
        out_prompts[s] = p

    save_json(OUT_PROMPTS, out_prompts)
    log(f"Done. Species scanned: {len(species)}. Updated with common names for {updated} species.")
    log(f"Wrote: {OUT_PROMPTS}, {CACHE_JSON}, {MAPPING_CSV}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0, help='If >0, only process the first N species (dry-run).')
    parser.add_argument('--english', action='store_true', help='If set, keep only English/Latin-script common names.')
    args = parser.parse_args()
    main(args)
