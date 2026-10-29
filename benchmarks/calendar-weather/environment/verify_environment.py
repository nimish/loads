"""Verify the reconstructed environment against the observed package inventory."""
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
records = json.loads((root / "installed-environment.json").read_text())["packages"]
expected = {item["name"]: item["version"] for item in records}
actual = {
    re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(): dist.version
    for dist in metadata.distributions()
}
assert sys.version_info[:3] == (3, 12, 6), sys.version
assert actual == expected, {
    "missing": sorted(expected.keys() - actual.keys()),
    "additional": sorted(actual.keys() - expected.keys()),
    "different": {name: (version, actual.get(name))
                  for name, version in expected.items() if actual.get(name) != version},
}
git_count = 0
for item in records:
    if "rev" not in item:
        continue
    direct = json.loads(metadata.distribution(item["name"]).read_text("direct_url.json"))
    assert direct["vcs_info"]["commit_id"] == item["rev"], item["name"]
    git_count += 1
print(f"Verified {len(expected)} package versions and {git_count} Git commits.")
