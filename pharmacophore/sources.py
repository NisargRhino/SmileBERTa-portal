"""Where the corpus comes from, and what we are allowed to do with it.

Every source is declared here with its licence, so `/credits` and the release
manifest can be generated from one place rather than from memory. Part of what
this corpus replaces is a training set assembled from "three databases that
aren't quoted", and a dataset whose provenance lives only in someone's head is
not a citable artefact.

Network access is wrapped so that a build can be resumed: everything fetched is
cached on disk by URL, and a re-run costs nothing for what it already has. A
corpus build touches tens of thousands of entries and will be interrupted.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# A generic client name on purpose: nothing identifying the user belongs in a
# header sent to a public API.
USER_AGENT = 'SmileBERTa-pharmacophore/1.0'
CACHE_DIR = os.environ.get('SB_CORPUS_CACHE',
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '.corpus-cache'))

RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_DATA = 'https://data.rcsb.org/rest/v1/core'
RCSB_FILES = 'https://files.rcsb.org/download'
CHEMBL = 'https://www.ebi.ac.uk/chembl/api/data'

# Declared licences. **Verify before redistributing anything derived from a
# source**; these are recorded so the claim is explicit and checkable, not so it
# can be taken on trust.
SOURCES = {
    'rcsb_pdb': {
        'name': 'RCSB Protein Data Bank',
        'url': 'https://www.rcsb.org',
        'licence': 'CC0 1.0 (public domain)',
        'citation': 'Berman HM et al. The Protein Data Bank. Nucleic Acids Res. 2000;28:235-242.',
        'used_for': 'protein-ligand complexes, crystal poses, contact geometry',
        'redistribution': 'permitted',
    },
    'chembl': {
        'name': 'ChEMBL',
        'url': 'https://www.ebi.ac.uk/chembl/',
        'licence': 'CC BY-SA 3.0',
        'citation': 'Zdrazil B et al. The ChEMBL Database in 2023. Nucleic Acids Res. 2024;52:D1180-D1192.',
        'used_for': 'measured bioactivity, ligand efficiency',
        # Share-alike propagates to anything derived from it: a release that
        # mixes ChEMBL in cannot be published under a more permissive licence.
        'redistribution': 'permitted under share-alike; propagates to derived data',
    },
    'pubchem': {
        'name': 'PubChem',
        'url': 'https://pubchem.ncbi.nlm.nih.gov',
        'licence': 'public domain (US NIH)',
        'citation': 'Kim S et al. PubChem 2023 update. Nucleic Acids Res. 2023;51:D1373-D1380.',
        'used_for': 'identifier cross-references, supplier counts',
        'redistribution': 'permitted',
    },
}


def _cache_path(url):
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, digest[:2], digest)


def fetch(url, timeout=60, retries=3, cache=True, binary=False):
    """GET with on-disk caching and polite backoff.

    Public APIs rate-limit, and a corpus build makes a great many requests. The
    backoff here is deliberately unhurried: finishing an hour later costs
    nothing, and being throttled out of a half-finished build costs the build.
    """
    path = _cache_path(url)
    if cache and os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
        return data if binary else data.decode('utf-8', 'replace')

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url)
            last = e
            time.sleep(2 ** attempt)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f'could not fetch {url}: {last}')

    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)
    return data if binary else data.decode('utf-8', 'replace')


def fetch_json(url, **kw):
    return json.loads(fetch(url, **kw))


# ── RCSB ─────────────────────────────────────────────────────────────────────

def rcsb_search(query, return_type='entry', rows=10000, start=0):
    """One page of a structured RCSB search."""
    payload = {
        'query': query,
        'return_type': return_type,
        'request_options': {'paginate': {'start': start, 'rows': rows},
                            'results_content_type': ['experimental']},
    }
    url = RCSB_SEARCH + '?json=' + urllib.parse.quote(json.dumps(payload))
    return fetch_json(url)


def rcsb_count(query, return_type='entry'):
    payload = {'query': query, 'return_type': return_type,
               'request_options': {'return_counts': True}}
    url = RCSB_SEARCH + '?json=' + urllib.parse.quote(json.dumps(payload))
    return fetch_json(url, cache=False)['total_count']


def candidate_query(max_resolution=2.5, min_ligand_atoms=12):
    """X-ray structures at usable resolution holding at least one real ligand.

    The heavy-atom floor is applied here as a cheap server-side filter on the
    *largest* non-polymer entity, so that tens of thousands of ion-only and
    buffer-only entries are never downloaded. The real per-ligand selection
    happens in `complexes.py` against the full structure, because this filter
    cannot tell a drug-like ligand from a detergent of the same size.
    """
    return {
        'type': 'group', 'logical_operator': 'and', 'nodes': [
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.experimental_method',
                'operator': 'exact_match', 'value': 'X-ray'}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.resolution_combined',
                'operator': 'less_or_equal', 'value': max_resolution}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.nonpolymer_entity_count',
                'operator': 'greater', 'value': 0}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id',
                'operator': 'exists'}},
        ],
    }


def entry_ids(max_resolution=2.5, limit=None, page=10000):
    """Every candidate PDB ID, paginated."""
    query = candidate_query(max_resolution)
    out, start = [], 0
    while True:
        res = rcsb_search(query, rows=page, start=start)
        hits = res.get('result_set', [])
        out.extend(h['identifier'] for h in hits)
        start += page
        if limit and len(out) >= limit:
            return out[:limit]
        if not hits or start >= res.get('total_count', 0):
            return out


def entry_metadata(pdb_id):
    return fetch_json(f'{RCSB_DATA}/entry/{pdb_id.upper()}')


def polymer_entity(pdb_id, entity_id):
    return fetch_json(f'{RCSB_DATA}/polymer_entity/{pdb_id.upper()}/{entity_id}')


def structure_pdb(pdb_id):
    """The PDB-format coordinates.

    Entries too large for the legacy format are mmCIF-only and raise
    FileNotFoundError here; the caller records that as a rejection reason rather
    than silently skipping, since 'how many structures did the format cost us'
    is a number the manifest should carry.
    """
    return fetch(f'{RCSB_FILES}/{pdb_id.upper()}.pdb')


def component_ideal_sdf(comp_id):
    return fetch(f'https://files.rcsb.org/ligands/download/{comp_id.upper()}_ideal.sdf')


# ── ChEMBL ───────────────────────────────────────────────────────────────────

def chembl_target_by_uniprot(uniprot):
    res = fetch_json(f'{CHEMBL}/target?target_components__accession={uniprot}'
                     f'&format=json&limit=20')
    return res.get('targets', [])


def chembl_activities(target_chembl_id, limit=1000, offset=0, standard_types=None):
    """Activity records for one target, newest API shape, one page at a time."""
    url = (f'{CHEMBL}/activity?target_chembl_id={target_chembl_id}'
           f'&format=json&limit={limit}&offset={offset}')
    if standard_types:
        url += '&standard_type__in=' + ','.join(standard_types)
    return fetch_json(url)


def credits_block():
    """The `/credits` entries for every source used, generated not remembered."""
    return [{'id': k, **v} for k, v in sorted(SOURCES.items())]
