# -*- coding: utf-8 -*-
"""
Run the code cells of every tutorial notebook up to (not including) the first cell that loads
data from disk. Those cells import the dependencies and set up the device, so this catches
missing packages and import errors without the licence-gated model files or AMASS data.
"""
import json
import os
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / 'notebooks'
DATA_LOAD = re.compile(r"np\.load\(|\.npz|torch\.load\(|open\(")


def notebook_prefix(path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if DATA_LOAD.search(source):
            break
        # drop IPython line magics, which are not Python
        source = '\n'.join(line for line in source.split('\n') if not line.lstrip().startswith(('%', '!')))
        cells.append(source)
    return cells


@pytest.mark.parametrize('notebook', sorted(p.name for p in NOTEBOOKS_DIR.glob('0*.ipynb')))
def test_notebook_prefix_runs(notebook, monkeypatch):
    cells = notebook_prefix(NOTEBOOKS_DIR / notebook)
    assert cells, f'{notebook}: no code cell before the first data load'
    monkeypatch.chdir(NOTEBOOKS_DIR)
    namespace = {'__name__': '__main__'}
    for index, source in enumerate(cells):
        exec(compile(source, f'{notebook}[cell {index}]', 'exec'), namespace)
