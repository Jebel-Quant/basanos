"""Run the README's ``pycon`` examples as doctests.

The README writes its examples as ``pycon`` transcripts — ``>>>`` prompts with
the expected output inline. ``pytest-rhiza``'s README checks only execute
``python`` fences, so without this test nothing notices when an example drifts
from what the library actually returns.

The fences are run in document order with shared globals, because the README
reads top to bottom: later examples reuse the names earlier ones define. They
run from a temporary working directory, since the reporting examples write
HTML files to relative ``output/`` paths.
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"

_PYCON_FENCE = re.compile(r"^```pycon\n(.*?)^```", re.MULTILINE | re.DOTALL)


def test_readme_pycon_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``pycon`` example in README.md produces its documented output."""
    monkeypatch.chdir(tmp_path)
    fences = _PYCON_FENCE.findall(README.read_text(encoding="utf-8"))
    assert fences, "README.md has no pycon fences to check"

    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs: dict[str, object] = {}
    for index, fence in enumerate(fences):
        test = parser.get_doctest(fence, globs, f"README.md[pycon {index}]", str(README), 0)
        runner.run(test, clear_globs=False)
        globs = test.globs

    results = runner.summarize(verbose=False)
    assert results.failed == 0, f"{results.failed} of {results.attempted} README examples failed"
