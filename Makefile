# Makefile for protein disentanglement experiments

.PHONY: help install test train evaluate clean docker aws-setup

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run quick test"
	@echo "  train       - Train all models"
	@echo "  evaluate    - Evaluate latest checkpoint"
	@echo "  clean       - Clean generated files"
	@echo "  docker      - Build Docker image"
	@echo "  aws-setup   - Setup AWS environment"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Installation complete!"

# Quick test
test:
	python train.py --debug --model beta_vae
	@echo "Test completed!"

# Training
train:
	python train.py --model all --gpu
	@echo "Training completed!"

train-beta:
	python train.py --model beta_vae --gpu

train-factor:
	python train.py --model factor_vae --gpu

train-simclr:
	python train.py --model simclr --gpu

# Evaluation  
evaluate:
	@if [ -f experiments/checkpoints/last.ckpt ]; then \
		python evaluate.py --checkpoint experiments/checkpoints/last.ckpt --plot; \
	else \
		echo "No checkpoint found. Run 'make train' first."; \
	fi

evaluate-all:
	@for ckpt in experiments/checkpoints/*.ckpt; do \
		if [ -f "$$ckpt" ]; then \
			echo "Evaluating $$ckpt"; \
			python evaluate.py --checkpoint "$$ckpt" --plot; \
		fi; \
	done

# Data preparation
data:
	mkdir -p data
	python -c "from src.data.utils import create_sample_fasta; create_sample_fasta('data/sample_proteins.fasta', 1000)"
	@echo "Sample data created!"

# Cleaning
clean:
	rm -rf experiments/*
	rm -rf logs/*
	rm -rf evaluation_results/*
	rm -rf __pycache__
	rm -rf src/*/__pycache__
	rm -rf src/*/*/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Cleaned generated files!"

clean-data:
	rm -rf data/*
	@echo "Cleaned data files!"

# Docker
docker:
	docker build -t protein-disentanglement .
	@echo "Docker image built!"

docker-run:
	docker run --gpus all -v $(PWD)/experiments:/app/experiments protein-disentanglement python train.py --model all

docker-shell:
	docker run --gpus all -it -v $(PWD):/app protein-disentanglement bash

# AWS
aws-setup:
	chmod +x aws_setup.sh
	./aws_setup.sh

# Development
dev-install:
	pip install -e .
	pip install jupyter notebook ipykernel
	python -m ipykernel install --user --name protein-disentanglement

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Monitoring
monitor:
	@echo "Monitoring training progress..."
	tail -f logs/training.log

gpu-status:
	nvidia-smi

# Quality checks
lint:
	flake8 src/ --max-line-length=88 --ignore=E203,W503
	black --check src/

format:
	black src/
	isort src/

# Testing
test-unit:
	python -m pytest tests/ -v

test-integration:
	python test_integration.py

# Backup
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		src/ config.yaml requirements.txt train.py evaluate.py \
		experiments/checkpoints/ experiments/*_results.pt

# Documentation
docs:
	mkdir -p docs
	python -m pydoc -w src.data.utils src.models.vae src.models.simclr src.evaluation.metrics
	mv *.html docs/
	@echo "Documentation generated in docs/"

# Experiment management
list-experiments:
	@echo "Available checkpoints:"
	@ls -la experiments/checkpoints/ || echo "No checkpoints found"
	@echo "\nExperiment results:"
	@ls -la experiments/*.pt || echo "No results found"

archive-experiment:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make archive-experiment NAME=experiment_name"; \
	else \
		mkdir -p archives/$(NAME); \
		cp -r experiments/* archives/$(NAME)/; \
		echo "Experiment archived as $(NAME)"; \
	fi

# Utility targets
check-gpu:
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

check-deps:
	python -c "import torch, pytorch_lightning, transformers, esm, Bio; print('All dependencies OK')"

version:
	@echo "Python version:"
	@python --version
	@echo "PyTorch version:"
	@python -c "import torch; print(torch.__version__)"
	@echo "CUDA version:"
	@python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'No CUDA')"
