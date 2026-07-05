"""Unified experiment runner for all model types.

Orchestrates data loading, preprocessing, embedding, model training,
evaluation, and result saving for classical, template quantum, and
custom VQC experiments.
"""

import logging
import time
import os
import json
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _safe_import_torch():
    """Safely import torch."""
    try:
        import torch
        return torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        raise


def run_classical_experiment(
    config,
    languages: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all classical baseline models.

    Parameters
    ----------
    config : Config
        Framework configuration.
    languages : list[str]
        Languages to run on.
    max_samples : int, optional
        Limit dataset size for debugging.

    Returns
    -------
    dict
        Results per language per model.
    """
    from data.loader import load_dataset_for_language
    from preprocessing.pipeline import preprocess_dataset
    from embeddings.factory import get_embedder, embed_dataset
    from evaluation.metrics import compute_metrics

    results = {}

    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Classical Experiment — {lang.upper()}")
        logger.info(f"{'='*60}")

        # 1. Load and preprocess
        dataset = load_dataset_for_language(lang, config, max_samples)
        dataset = preprocess_dataset(dataset, config)

        # 2. Embed
        embedder = get_embedder(config, lang)
        embeddings = embed_dataset(embedder, dataset)

        X_train = embeddings["train"]
        y_train = np.array(dataset.train["labels"])
        X_val = embeddings["val"]
        y_val = np.array(dataset.val["labels"])
        X_test = embeddings["test"]
        y_test = np.array(dataset.test["labels"])

        input_dim = X_train.shape[1]
        num_labels = dataset.num_labels
        device = config.get("device", "cpu")

        lang_results = {}

        # 3. Train models
        models_to_run = {
            "logistic_regression": _run_logreg,
            "svm": _run_svm,
            "cnn": _run_cnn,
            "bilstm": _run_bilstm,
        }

        for model_name, run_fn in models_to_run.items():
            try:
                logger.info(f"\n--- Training {model_name} ---")
                model_results = run_fn(
                    config, X_train, y_train, X_val, y_val,
                    X_test, y_test, num_labels, input_dim, device,
                )
                lang_results[model_name] = model_results
                logger.info(f"  {model_name}: F1={model_results['test_metrics']['f1']:.4f}")
            except Exception as e:
                logger.error(f"  {model_name} failed: {e}")
                lang_results[model_name] = {"error": str(e)}

        results[lang] = lang_results

    _save_experiment_results(results, "classical", config)
    return results


def _run_logreg(config, X_train, y_train, X_val, y_val, X_test, y_test,
                num_labels, input_dim, device):
    from models.classical.logistic_regression import LogisticRegressionClassifier
    clf = LogisticRegressionClassifier(config, num_labels)
    history = clf.train(X_train, y_train, X_val, y_val)
    test_metrics = clf.evaluate(X_test, y_test)
    return {"history": history, "test_metrics": test_metrics,
            "params": clf.get_params_count()}


def _run_svm(config, X_train, y_train, X_val, y_val, X_test, y_test,
             num_labels, input_dim, device):
    from models.classical.svm import SVMClassifier
    clf = SVMClassifier(config, num_labels)
    history = clf.train(X_train, y_train, X_val, y_val)
    test_metrics = clf.evaluate(X_test, y_test)
    return {"history": history, "test_metrics": test_metrics,
            "params": clf.get_params_count()}


def _run_cnn(config, X_train, y_train, X_val, y_val, X_test, y_test,
             num_labels, input_dim, device):
    from models.classical.cnn import CNNClassifier
    clf = CNNClassifier(config, num_labels, device)
    history = clf.train(X_train, y_train, X_val, y_val)
    test_metrics = clf.evaluate(X_test, y_test)
    return {"history": history, "test_metrics": test_metrics,
            "params": clf.get_params_count()}


def _run_bilstm(config, X_train, y_train, X_val, y_val, X_test, y_test,
                num_labels, input_dim, device):
    from models.classical.bilstm import BiLSTMClassifier
    clf = BiLSTMClassifier(config, num_labels, device)
    history = clf.train(X_train, y_train, X_val, y_val)
    test_metrics = clf.evaluate(X_test, y_test)
    return {"history": history, "test_metrics": test_metrics,
            "params": clf.get_params_count()}


def run_quantum_template_experiment(
    config,
    languages: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run template-based quantum circuit experiments."""
    from data.loader import load_dataset_for_language
    from preprocessing.pipeline import preprocess_dataset
    from embeddings.factory import get_embedder, embed_dataset
    from models.quantum_template.hybrid_model import QuantumTemplateClassifier

    results = {}

    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Quantum Template Experiment — {lang.upper()}")
        logger.info(f"{'='*60}")

        dataset = load_dataset_for_language(lang, config, max_samples)
        dataset = preprocess_dataset(dataset, config)
        embedder = get_embedder(config, lang)
        embeddings = embed_dataset(embedder, dataset)

        X_train = embeddings["train"]
        y_train = np.array(dataset.train["labels"])
        X_val = embeddings["val"]
        y_val = np.array(dataset.val["labels"])
        X_test = embeddings["test"]
        y_test = np.array(dataset.test["labels"])

        input_dim = X_train.shape[1]
        num_labels = dataset.num_labels
        device = config.get("device", "cpu")

        # Get circuit types to test
        circuit_type = getattr(config.quantum_template, "circuit_type",
                               "strongly_entangling")
        circuit_types = [circuit_type] if isinstance(circuit_type, str) else circuit_type

        lang_results = {}
        for ct in circuit_types:
            try:
                logger.info(f"\n--- Training Quantum Template: {ct} ---")
                logger.info(f"    Qubits: {config.quantum_template.n_qubits}, "
                            f"Layers: {config.quantum_template.n_layers}")

                clf = QuantumTemplateClassifier(
                    config, num_labels, input_dim, device
                )
                # Override circuit type if testing multiple
                clf.circuit_type = ct

                history = clf.train(X_train, y_train, X_val, y_val)
                test_metrics = clf.evaluate(X_test, y_test)
                lang_results[ct] = {
                    "history": history,
                    "test_metrics": test_metrics,
                    "params": clf.get_params_count(),
                    "n_qubits": config.quantum_template.n_qubits,
                    "n_layers": config.quantum_template.n_layers,
                }
                logger.info(f"  {ct}: F1={test_metrics['f1']:.4f}")
            except Exception as e:
                logger.error(f"  {ct} failed: {e}")
                lang_results[ct] = {"error": str(e)}

        results[lang] = lang_results

    _save_experiment_results(results, "quantum_template", config)
    return results


def run_custom_vqc_experiment(
    config,
    languages: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run custom VQC experiments (v1, v2, v3)."""
    from data.loader import load_dataset_for_language
    from preprocessing.pipeline import preprocess_dataset
    from embeddings.factory import get_embedder, embed_dataset
    from models.custom_vqc.hybrid_model import CustomVQCClassifier

    results = {}

    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Custom VQC Experiment — {lang.upper()}")
        logger.info(f"{'='*60}")

        dataset = load_dataset_for_language(lang, config, max_samples)
        dataset = preprocess_dataset(dataset, config)
        embedder = get_embedder(config, lang)
        embeddings = embed_dataset(embedder, dataset)

        X_train = embeddings["train"]
        y_train = np.array(dataset.train["labels"])
        X_val = embeddings["val"]
        y_val = np.array(dataset.val["labels"])
        X_test = embeddings["test"]
        y_test = np.array(dataset.test["labels"])

        input_dim = X_train.shape[1]
        num_labels = dataset.num_labels
        device = config.get("device", "cpu")

        version = getattr(config.custom_vqc, "version", "v1")
        versions = [version] if isinstance(version, str) else version

        lang_results = {}
        for ver in versions:
            try:
                logger.info(f"\n--- Training Custom VQC {ver} ---")
                logger.info(f"    Qubits: {config.custom_vqc.n_qubits}, "
                            f"Layers: {config.custom_vqc.n_layers}")

                # Temporarily set version in config
                original_version = config.custom_vqc.version
                config.custom_vqc.version = ver

                clf = CustomVQCClassifier(
                    config, num_labels, input_dim, device
                )

                history = clf.train(X_train, y_train, X_val, y_val)
                test_metrics = clf.evaluate(X_test, y_test)

                lang_results[ver] = {
                    "history": history,
                    "test_metrics": test_metrics,
                    "params": clf.get_params_count(),
                    "n_qubits": config.custom_vqc.n_qubits,
                    "n_layers": config.custom_vqc.n_layers,
                    "version": ver,
                }
                logger.info(f"  {ver}: F1={test_metrics['f1']:.4f}")

                config.custom_vqc.version = original_version

            except Exception as e:
                logger.error(f"  {ver} failed: {e}")
                lang_results[ver] = {"error": str(e)}

        results[lang] = lang_results

    _save_experiment_results(results, "custom_vqc", config)
    return results


def run_all_experiments(
    config,
    languages: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all experiment types sequentially."""
    all_results = {}

    logger.info("\n" + "=" * 70)
    logger.info("  RUNNING ALL EXPERIMENTS")
    logger.info("=" * 70)

    # Classical
    try:
        all_results["classical"] = run_classical_experiment(
            config, languages, max_samples
        )
    except Exception as e:
        logger.error(f"Classical experiments failed: {e}")
        all_results["classical"] = {"error": str(e)}

    # Quantum Template
    try:
        all_results["quantum_template"] = run_quantum_template_experiment(
            config, languages, max_samples
        )
    except Exception as e:
        logger.error(f"Quantum template experiments failed: {e}")
        all_results["quantum_template"] = {"error": str(e)}

    # Custom VQC
    try:
        all_results["custom_vqc"] = run_custom_vqc_experiment(
            config, languages, max_samples
        )
    except Exception as e:
        logger.error(f"Custom VQC experiments failed: {e}")
        all_results["custom_vqc"] = {"error": str(e)}

    _save_experiment_results(all_results, "all_experiments", config)
    _print_summary(all_results)
    return all_results


def _save_experiment_results(results: dict, name: str, config) -> str:
    """Save results to JSON file."""
    output_dir = getattr(config, "output_dir", "results")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"{name}_{timestamp}.json")

    # Make results JSON-serializable
    serializable = _make_serializable(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {filepath}")
    return filepath


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _print_summary(all_results: dict) -> None:
    """Print a summary table of all results."""
    logger.info("\n" + "=" * 70)
    logger.info("  EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    for exp_type, exp_results in all_results.items():
        if isinstance(exp_results, dict) and "error" not in exp_results:
            logger.info(f"\n  {exp_type.upper()}:")
            for lang, lang_results in exp_results.items():
                if isinstance(lang_results, dict):
                    for model, model_results in lang_results.items():
                        if isinstance(model_results, dict) and "test_metrics" in model_results:
                            m = model_results["test_metrics"]
                            logger.info(
                                f"    {lang}/{model}: "
                                f"Acc={m.get('accuracy', 0):.4f} "
                                f"F1={m.get('f1', 0):.4f} "
                                f"Params={model_results.get('params', '?')}"
                            )
