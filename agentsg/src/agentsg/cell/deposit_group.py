"""PDB deposition-group ids, stored beside the UniProt list.

RCSB records a group deposition as ``rcsb_entry_group_membership`` with
``aggregation_method`` ``matching_deposit_group_id``. The id looks like
``G_1002344``. The label is ``DEPOSITION_GROUP_<id>``. An entry may carry
both this id and a UniProt accession.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
DEPOSIT_GROUP_METHOD = "matching_deposit_group_id"

_QUERY = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_group_membership {
      aggregation_method
      group_id
    }
  }
}
"""


def deposition_group_label(group_id):
    """``DEPOSITION_GROUP_<id>`` for a PDB deposition-group id."""
    return f"DEPOSITION_GROUP_{group_id}"


def pick_deposit_group(memberships):
    """The deposition-group id on one entry, or ``""`` when it has none.

    Other group memberships (UniProt, sequence clusters) are ignored. Two
    deposition groups on one entry resolve to the lexicographically first id.
    """
    if not memberships:
        return ""
    ids = sorted({
        str(item["group_id"])
        for item in memberships
        if item
        and item.get("aggregation_method") == DEPOSIT_GROUP_METHOD
        and item.get("group_id")
    })
    return ids[0] if ids else ""


def deposit_groups_from_entries(entries, wanted):
    """Map each wanted PDB id to its deposition-group id, or ``""``."""
    found = {}
    for entry in entries or []:
        if not entry or not entry.get("rcsb_id"):
            continue
        pid = str(entry["rcsb_id"]).upper()
        found[pid] = pick_deposit_group(entry.get("rcsb_entry_group_membership"))
    return {pid: found.get(pid, "") for pid in wanted}


def fetch_deposit_groups(pdb_ids, *, batch=200, progress=False):
    """Ask RCSB for the deposition group of each id.

    Returns ``{PDB_ID: group_id or ""}``. An id RCSB does not return is
    ``""``. Requests are retried on a transient HTTP failure.
    """
    ids = [str(pid).upper() for pid in pdb_ids]
    out = {}
    total = len(ids)
    for start in range(0, total, batch):
        chunk = ids[start:start + batch]
        payload = _graphql({"query": _QUERY, "variables": {"ids": chunk}})
        if payload.get("errors"):
            message = payload["errors"][0].get("message", "GraphQL error")
            raise RuntimeError(f"RCSB GraphQL {RCSB_GRAPHQL}: {message}")
        data = payload.get("data") or {}
        out.update(deposit_groups_from_entries(data.get("entries"), chunk))
        if progress:
            print(f"  deposition groups {min(start + batch, total):,}/{total:,}", flush=True)
    return out


def _graphql(payload):
    body = json.dumps(payload).encode()
    delay = 1.0
    last = None
    for attempt in range(5):
        request = urllib.request.Request(
            RCSB_GRAPHQL,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"RCSB GraphQL {RCSB_GRAPHQL} returned {exc.code}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < 4:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"RCSB GraphQL {RCSB_GRAPHQL} failed") from last
