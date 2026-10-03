# ORBIT v0.1.0 release checklist

This repository is prepared as the software companion to **ORBIT: Function-Space Optimization for Rotary Query-Key Interactions**.

Before making the repository public and creating the immutable release:

1. confirm the final paper PDF points to `https://github.com/pop123-ux/ORBIT`;
2. confirm `main` CI is green on Python 3.10, 3.11, 3.12 and the minimum supported PyTorch line;
3. verify `configs/paper_124m_matched.json` against the manuscript's frozen primary recipe and seeds;
4. run `pytest -q` from a fresh clone;
5. run `python examples/quickstart.py`;
6. make the repository public;
7. tag the exact reviewed commit as `v0.1.0` and create the GitHub Release from the `0.1.0` changelog entry.

Recommended local release commands after the public-state switch:

```bash
git clone https://github.com/pop123-ux/ORBIT.git
cd ORBIT
git checkout main
git pull --ff-only
pytest -q
python examples/quickstart.py

git tag -a v0.1.0 -m "ORBIT v0.1.0"
git push origin v0.1.0
```

Then create the GitHub Release for `v0.1.0` using the `CHANGELOG.md` entry as the release notes.

The tag should be created only after the repository is public and the final commit has been manually reviewed, so that the published paper, repository, and immutable software snapshot all refer to the same implementation.
