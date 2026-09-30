.PHONY: test smoke evidence paper

test:
	pytest -q

smoke:
	python examples/quickstart.py

evidence:
	rm -rf .orbit-frozen-work
	mkdir -p .orbit-frozen-work/merged
	cp results/paper-v1/raw/*.jsonl .orbit-frozen-work/merged/
	cp configs/matched_best_configs.json .orbit-frozen-work/merged/
	python scripts/orbit_paper_artifacts.py --work-dir .orbit-frozen-work

paper:
	python scripts/rebuild_frozen_paper.py
