# SPDX-License-Identifier: Apache-2.0
"""Fail an image build when the tested source overlays are not present."""
import hashlib
import json
from pathlib import Path

root = Path('/opt/raon/native-fusion')
site = Path('/usr/local/lib/python3.14/dist-packages')
expected = json.loads((root / 'tools/raon_validation/results/native-stack-20261128/tested-source-sha256.json').read_bytes())
for name, digest in expected.items():
    path = (site if name.startswith('vllm/') else root) / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
package = site / 'vllm/models/raon1_mini'
manifest = {
    str(path.relative_to(site)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(package.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts
}
(root / 'installed-model-source-sha256.json').write_text(json.dumps(manifest, indent=2))
print('VALIDATED_NATIVE_STACK_IMAGE_SOURCE', len(expected), len(manifest), flush=True)
