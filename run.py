#!/usr/bin/env python3
"""
Hybrid Quantum-Classical NLP Framework — CLI Entry Point
========================================================

Run experiments, generate visualizations, and produce result tables
for multilingual emotion detection using classical, template quantum,
and custom VQC models.

Usage Examples
--------------
# Classical baselines
python run.py --experiment classical --language english --embedding tfidf

# Template quantum model
python run.py --experiment quantum_template --language english --qubits 4

# Custom VQC
python run.py --experiment custom_vqc --version v2 --language english --qubits 6

# Low-resource learning
python run.py --experiment low_resource --fraction 0.1 --language english

# Run all experiments
python run.py --experiment all

# Quick test (100 samples only)
python run.py --experiment classical --language english --max-samples 100
"""

import argparse
import logging
import os
import sys
import time
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, save_config


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO", log_dir: str = "results/logs") -> None:
    """Configure root logger with console + file handlers."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    fmt = "[%(asctime)s] %(levelname)-8s %(name)-30s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logging.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Hybrid Quantum-Classical NLP Framework for Emotion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Experiment selection
    parser.add_argument(
        "--experiment", type=str, default=None,
        choices=["classical", "quantum_template", "custom_vqc",
                 "low_resource", "all"],
        help="Which experiment to run.",
    )

    # -- Language
    parser.add_argument(
        "--language", type=str, default="english",
        choices=["english", "hindi", "bengali", "all"],
        help="Target language (default: english).",
    )

    # -- Embedding
    parser.add_argument(
        "--embedding", type=str, default=None,
        choices=["tfidf", "fasttext", "word2vec", "bert", "mbert", "indicbert"],
        help="Embedding method (overrides config).",
    )

    # -- Quantum-specific
    parser.add_argument("--qubits", type=int, default=None, choices=[4, 6, 8])
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--circuit", type=str, default=None,
        choices=["strongly_entangling", "basic_entangler", "efficient_su2"],
    )
    parser.add_argument(
        "--version", type=str, default=None, choices=["v1", "v2", "v3"],
    )
    parser.add_argument(
        "--entanglement", type=str, default=None, choices=["linear", "ring"],
    )

    # -- Low-resource
    parser.add_argument("--fraction", type=float, default=None)

    # -- General
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for quick testing).")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # -- Config
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--exp-config", type=str, default=None)

    return parser


# ---------------------------------------------------------------------------
# Apply CLI overrides to config
# ---------------------------------------------------------------------------
def apply_cli_overrides(args: argparse.Namespace) -> dict:
    """Apply CLI arguments as overrides on top of the config dict."""
    overrides = {}

    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.embedding is not None:
        overrides.setdefault("embeddings", {})["method"] = args.embedding
    if args.qubits is not None:
        overrides.setdefault("quantum_template", {})["n_qubits"] = args.qubits
        overrides.setdefault("custom_vqc", {})["n_qubits"] = args.qubits
    if args.layers is not None:
        overrides.setdefault("quantum_template", {})["n_layers"] = args.layers
        overrides.setdefault("custom_vqc", {})["n_layers"] = args.layers
    if args.circuit is not None:
        overrides.setdefault("quantum_template", {})["circuit_type"] = args.circuit
    if args.version is not None:
        overrides.setdefault("custom_vqc", {})["version"] = args.version
    if args.entanglement is not None:
        overrides.setdefault("custom_vqc", {})["entanglement"] = args.entanglement
    if args.fraction is not None:
        overrides.setdefault("low_resource", {})["fractions"] = [args.fraction]
        overrides.setdefault("low_resource", {})["enabled"] = True

    return overrides


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def run_experiment(experiment: str, language: str, cfg, max_samples=None):
    """Dispatch to the appropriate experiment runner."""
    from experiments.runner import (
        run_classical_experiment,
        run_quantum_template_experiment,
        run_custom_vqc_experiment,
        run_all_experiments,
    )

    languages = (
        ["english", "hindi", "bengali"] if language == "all" else [language]
    )

    logger = logging.getLogger("run")
    logger.info("=" * 70)
    logger.info(f"  EXPERIMENT : {experiment}")
    logger.info(f"  LANGUAGES  : {languages}")
    logger.info(f"  EMBEDDING  : {cfg.embeddings.method}")
    if max_samples:
        logger.info(f"  MAX SAMPLES: {max_samples}")
    logger.info("=" * 70)

    if experiment == "classical":
        return run_classical_experiment(cfg, languages, max_samples)

    elif experiment == "quantum_template":
        logger.info(f"  Circuit: {cfg.quantum_template.circuit_type}")
        logger.info(f"  Qubits:  {cfg.quantum_template.n_qubits}")
        logger.info(f"  Layers:  {cfg.quantum_template.n_layers}")
        return run_quantum_template_experiment(cfg, languages, max_samples)

    elif experiment == "custom_vqc":
        logger.info(f"  Version:      {cfg.custom_vqc.version}")
        logger.info(f"  Qubits:       {cfg.custom_vqc.n_qubits}")
        logger.info(f"  Layers:       {cfg.custom_vqc.n_layers}")
        logger.info(f"  Entanglement: {cfg.custom_vqc.entanglement}")
        return run_custom_vqc_experiment(cfg, languages, max_samples)

    elif experiment == "all":
        return run_all_experiments(cfg, languages, max_samples)

    else:
        logger.error(f"Unknown experiment: {experiment}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    overrides = apply_cli_overrides(args)
    cfg = load_config(experiment=args.exp_config, overrides=overrides)

    # Setup
    log_dir = "results/logs"
    if hasattr(cfg, "logging") and hasattr(cfg.logging, "log_dir"):
        log_dir = cfg.logging.log_dir

    setup_logging(
        level=cfg.get("log_level", "INFO"),
        log_dir=log_dir,
    )
    set_seed(cfg.get("seed", 42))

    logger = logging.getLogger("run")
    logger.info("Hybrid Quantum-Classical NLP Framework")
    logger.info(f"Project root: {cfg.get('project_root', 'unknown')}")

    # Auto-detect GPU in Colab
    try:
        import torch
        if torch.cuda.is_available() and cfg.get("device", "cpu") == "cpu":
            logger.info("GPU detected! Consider using --device cuda for faster training.")
    except ImportError:
        pass

    # Dispatch
    if args.experiment:
        run_experiment(args.experiment, args.language, cfg, args.max_samples)
    else:
        parser.print_help()
        sys.exit(0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
