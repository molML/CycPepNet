PYTHON ?= python
SHELL := /bin/bash

.PHONY: \
	process-cycpeptmpdb process-macrocycledb process-npmmpd process-peplife2 process-pepmsnd process-peptherdia process-all \
	combine-cycpeptmpdb-npmmpd combine-halflife-peptides combine-all generate-splits \
	analyze-similarity analyze-tasks analyze-splits analyze-uncertainty analyze-tests analyze-ranking analyze-results analyze-distributions analyze-all \
	prepare-all run-framework-non-neural run-framework-neural run-framework \
	prepare reproduce verify

process-cycpeptmpdb:
	$(PYTHON) -m cycpepnet.process.process_cycpeptmpdb

process-macrocycledb:
	$(PYTHON) -m cycpepnet.process.process_macrocycledb

process-npmmpd:
	$(PYTHON) -m cycpepnet.process.process_npmmpd

process-peplife2:
	$(PYTHON) -m cycpepnet.process.process_peplife2

process-pepmsnd:
	$(PYTHON) -m cycpepnet.process.process_pepmsnd

process-peptherdia:
	$(PYTHON) -m cycpepnet.process.process_peptherdia

process-all: process-cycpeptmpdb process-macrocycledb process-npmmpd process-peplife2 process-pepmsnd process-peptherdia

combine-cycpeptmpdb-npmmpd: process-cycpeptmpdb process-npmmpd
	$(PYTHON) -m cycpepnet.combine.combine_cycpeptmpdb_npmmpd

combine-halflife-peptides: process-peplife2 process-pepmsnd process-peptherdia
	$(PYTHON) -m cycpepnet.combine.combine_halflife_peptides

combine-all: combine-cycpeptmpdb-npmmpd combine-halflife-peptides

generate-splits:
	$(PYTHON) -m cycpepnet.benchmark

analyze-similarity:
	$(PYTHON) -m cycpepnet.analyze.analyze_similarity

analyze-tasks:
	$(PYTHON) -m cycpepnet.analyze.analyze_tasks

analyze-splits:
	$(PYTHON) -m cycpepnet.analyze.analyze_splits

analyze-uncertainty:
	$(PYTHON) -m cycpepnet.analyze.analyze_uncertainty

analyze-tests: analyze-uncertainty
	$(PYTHON) -m cycpepnet.analyze.statistical_tests

analyze-ranking:
	$(PYTHON) -m cycpepnet.analyze.analyze_ranking

analyze-results: analyze-tests analyze-ranking
	$(PYTHON) -m cycpepnet.analyze.analyze_results

analyze-distributions: analyze-tests
	$(PYTHON) -m cycpepnet.analyze.analyze_distributions

analyze-all: reproduce

prepare-all:
	$(PYTHON) -m cycpepnet.prepare --stage all

run-framework-non-neural:
	$(PYTHON) -m cycpepnet.runner --model-type non-neural

run-framework-neural:
	$(PYTHON) -m cycpepnet.runner --model-type neural

run-framework:
	$(PYTHON) -m cycpepnet.runner

reproduce:
	$(PYTHON) -m cycpepnet.reproduce

verify:
	$(PYTHON) -m cycpepnet.verify

prepare:
	$(PYTHON) -m cycpepnet.prepare
