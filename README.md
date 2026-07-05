<p align="center">
  <h1 align="center">⚛️ Hybrid Quantum-Classical NLP Framework<br/>for Multilingual Emotion Detection</h1>
  <p align="center">
    <strong>Comparing Classical, Template Quantum, and Custom VQC Models<br/>for Emotion Classification in Bengali, Hindi, and English</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python" alt="Python 3.9+"/>
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
    <img src="https://img.shields.io/badge/framework-PennyLane-blueviolet?style=flat-square" alt="PennyLane"/>
    <img src="https://img.shields.io/badge/status-Active%20Research-orange?style=flat-square" alt="Status"/>
    <img src="https://img.shields.io/badge/repo-Private-red?style=flat-square" alt="Private"/>
  </p>
</p>

---

## 📋 Project Overview

This framework provides a comprehensive research platform for comparing **classical machine learning**, **hybrid quantum-classical models using PennyLane templates**, and **custom Variational Quantum Circuit (VQC) models** for emotion detection across three languages: **Bengali**, **Hindi**, and **English**.

The project investigates whether quantum-enhanced NLP models can outperform their classical counterparts, particularly in low-resource multilingual settings. It includes data augmentation strategies, memory-augmented architectures, and a full evaluation pipeline for reproducible experiments.

---

## 🎯 Research Motivation

Emotion detection in natural language is a critical task for affective computing, social media analysis, and mental health monitoring. While deep learning has achieved strong results for resource-rich languages like English, **low-resource languages** (Bengali, Hindi) remain underserved. This project explores a novel research direction:

> *Can hybrid quantum-classical models leverage the expressiveness of parameterized quantum circuits to achieve competitive or superior performance in multilingual emotion detection — especially when training data is scarce?*

### Research Questions

1. **RQ1:** Can hybrid quantum-classical NLP models improve emotion detection accuracy compared to purely classical approaches?
2. **RQ2:** How does a custom-designed VQC compare to PennyLane template circuits (e.g., `StronglyEntanglingLayers`) in terms of accuracy, F1 score, and training efficiency?
3. **RQ3:** Can quantum models achieve comparable performance with fewer training samples (sample efficiency)?
4. **RQ4:** Do quantum models generalize better than classical models in low-resource language settings?
5. **RQ5:** Can data augmentation and memory-augmented approaches further improve performance in low-resource scenarios?

---

## ✨ Features

- **Multi-paradigm comparison**: Classical ML, deep learning, template quantum, and custom VQC models
- **Multilingual support**: English, Hindi, and Bengali with language-specific preprocessing
- **Custom VQC architectures**: Three novel circuit designs (v1, v2, v3) with data re-uploading and ring entanglement
- **Low-resource evaluation**: Systematic training with 1%, 5%, 10%, 25%, and 50% of data
- **Modular pipeline**: Configurable via YAML configs with CLI overrides
- **Multiple embedding strategies**: TF-IDF, FastText, Word2Vec, BERT, mBERT, IndicBERT
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1 with statistical significance tests
- **Reproducible experiments**: Seed control, configuration logging, and result tracking

---

## 🏗️ Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                         │
│  │ English │  │  Hindi  │  │ Bengali │   Multilingual Datasets  │
│  └────┬────┘  └────┬────┘  └────┬────┘                         │
│       └─────────┬──┴───────────┘                                │
├─────────────────┼───────────────────────────────────────────────┤
│           PREPROCESSING PIPELINE                                 │
│  Unicode Norm → Text Cleaning → Tokenization → Label Mapping    │
├─────────────────┼───────────────────────────────────────────────┤
│           EMBEDDING LAYER                                        │
│  ┌───────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌────────┐ │
│  │TF-IDF │ │FastText  │ │W2Vec │ │ BERT │ │ mBERT │ │IndicBT │ │
│  └───┬───┘ └────┬────┘ └──┬───┘ └──┬───┘ └───┬───┘ └───┬────┘ │
│      └──────────┼─────────┼────────┼─────────┼─────────┘       │
│                 ▼         ▼        ▼         ▼                  │
├─────────────────────────────────────────────────────────────────┤
│                     MODEL LAYER                                  │
│                                                                  │
│  Classical Branch          Quantum Branch                        │
│  ┌────┐ ┌─────┐           ┌─────────────────────────┐          │
│  │ LR │ │ SVM │           │   Feature Compressor     │          │
│  └────┘ └─────┘           │   (Linear / PCA → n_q)   │          │
│  ┌─────┐ ┌───────┐       └───────────┬─────────────┘          │
│  │ CNN │ │BiLSTM │                   ▼                          │
│  └─────┘ └───────┘       ┌─────────────────────────┐          │
│                           │   Quantum Circuit (VQC)  │          │
│                           │  ┌────────┐ ┌─────────┐ │          │
│                           │  │Template│ │Custom   │ │          │
│                           │  │Circuits│ │v1/v2/v3 │ │          │
│                           │  └────────┘ └─────────┘ │          │
│                           └───────────┬─────────────┘          │
│                                       ▼                          │
│                           ┌─────────────────────────┐          │
│                           │  Post-Quantum Classifier │          │
│                           └─────────────────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                    EVALUATION LAYER                               │
│  Accuracy │ Precision │ Recall │ F1 │ Statistical Tests          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
Hybrid-Quantum-Multilingual-Emotion-Detection/
│
├── config/                          # Configuration management
│   ├── __init__.py                  #   Config loader & merger
│   ├── default.yaml                 #   Master configuration file
│   └── experiments/                 #   Per-experiment YAML overrides
│       ├── classical_baseline.yaml
│       ├── quantum_template.yaml
│       ├── custom_vqc.yaml
│       ├── low_resource.yaml
│       └── augmentation.yaml
│
├── data/                            # Data loading & processing
│   ├── __init__.py
│   ├── loader.py                    #   Dataset loading (HuggingFace, local, GitHub)
│   ├── registry.py                  #   Dataset registry & metadata
│   ├── splitter.py                  #   Train/val/test splitting
│   ├── raw/                         #   Raw dataset files
│   ├── processed/                   #   Preprocessed data cache
│   └── augmented/                   #   Augmented data cache
│
├── preprocessing/                   # Text preprocessing pipeline
│   ├── __init__.py
│   ├── cleaner.py                   #   Text cleaning (URLs, HTML, special chars)
│   ├── tokenizer.py                 #   Language-aware tokenization
│   ├── label_normalizer.py          #   Unified emotion label mapping
│   ├── unicode_normalizer.py        #   Unicode normalization for Indic scripts
│   └── pipeline.py                  #   End-to-end preprocessing orchestration
│
├── embeddings/                      # Feature extraction
│   ├── __init__.py
│   ├── tfidf.py                     #   TF-IDF vectorization
│   ├── bert_emb.py                  #   BERT/mBERT/IndicBERT embeddings
│   └── factory.py                   #   Embedding method factory
│
├── models/                          # Model implementations
│   ├── __init__.py
│   ├── classical/                   #   Classical ML & DL baselines
│   │   ├── base.py                  #     Abstract base class
│   │   ├── logistic_regression.py   #     Logistic Regression
│   │   ├── svm.py                   #     Support Vector Machine
│   │   ├── cnn.py                   #     1D CNN text classifier
│   │   └── bilstm.py               #     Bidirectional LSTM with attention
│   ├── quantum_template/            #   PennyLane template-based quantum models
│   │   ├── circuits.py              #     Template circuit definitions
│   │   ├── feature_compressor.py    #     Dimensionality reduction to n_qubits
│   │   └── hybrid_model.py          #     Hybrid classical-quantum model
│   ├── custom_vqc/                  #   Custom VQC architectures
│   │   ├── custom_vqc_v1.py         #     Basic VQC (rotation + entanglement)
│   │   ├── custom_vqc_v2.py         #     Data re-uploading VQC
│   │   ├── custom_vqc_v3.py         #     Ring entanglement VQC
│   │   ├── feature_compressor.py    #     Feature compression layer
│   │   └── hybrid_model.py          #     Hybrid classical-quantum model
│   └── memory/                      #   Memory-augmented variants (planned)
│
├── augmentation/                    # Data augmentation strategies
│   └── __init__.py                  #   Back-translation, synonym, contextual
│
├── experiments/                     # Experiment orchestration
│   ├── __init__.py
│   └── runner.py                    #   Experiment dispatcher & result logging
│
├── evaluation/                      # Metrics & analysis
│   ├── __init__.py
│   └── metrics.py                   #   Accuracy, F1, statistical tests
│
├── visualization/                   # Plotting & reporting
│   ├── __init__.py
│   └── plots.py                     #   Confusion matrices, comparisons, bar charts
│
├── results/                         # Experiment outputs
│   ├── figures/                     #   Generated plots
│   ├── tables/                      #   Result tables
│   └── logs/                        #   Experiment logs & JSON results
│
├── notebooks/                       # Jupyter notebooks
│   ├── 04_embedding_study.ipynb
│   ├── 05_vqc_ablation_study.ipynb
│   ├── 06_activation_ablation.ipynb
│   ├── 07_optimizer_ablation.ipynb
│   ├── 08_low_resource_learning.ipynb
│   ├── Hybrid_Quantum_NLP_Colab_FIXED_v2.ipynb
│   └── VQC_Small_Dataset_Training_FIXED.ipynb
│
├── run.py                           # CLI entry point
├── setup.py                         # Package setup script
├── utils.py                         # Shared utilities
├── test_datasets.py                 # Dataset integration tests
├── test_phase2.py                   # Phase 2 validation tests
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

```bash
# Clone the repository
git clone https://github.com/Soham-Santra/Hybrid-Quantum-Classical-NLP.git
cd Hybrid-Quantum-Classical-NLP

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch (select appropriate CUDA version from https://pytorch.org)
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Google Colab

For Colab users, `torch` and `torchvision` are pre-installed. Simply run:

```python
!pip install -q pennylane transformers datasets scikit-learn gensim nltk indic-nlp-library
```

---

## 📦 Dependencies

| Category | Packages |
|----------|----------|
| **Deep Learning** | PyTorch, Torchvision |
| **Quantum Computing** | PennyLane |
| **NLP & Transformers** | Transformers, Tokenizers, NLTK, Gensim |
| **Classical ML** | scikit-learn |
| **Indic NLP** | indic-nlp-library |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Configuration** | PyYAML, tqdm, tabulate |

See [requirements.txt](requirements.txt) for the full list with version constraints.

---

## ⚡ Running Experiments

### CLI Usage

```bash
# Classical model baselines
python run.py --experiment classical --language english --embedding tfidf

# Quantum template model
python run.py --experiment quantum_template --language bengali --qubits 4

# Custom VQC experiment
python run.py --experiment custom_vqc --version v2 --language hindi

# Low-resource experiment
python run.py --experiment low_resource --fraction 0.1

# Run all experiments
python run.py --experiment all

# Quick test (limited samples)
python run.py --experiment classical --language english --max-samples 100
```

### Configuration

Experiments are configured via YAML files in `config/`:

- **`config/default.yaml`** — Master configuration (datasets, models, hyperparameters)
- **`config/experiments/`** — Per-experiment overrides

CLI arguments override YAML settings. See `python run.py --help` for all options.

---

## 🔬 Experimental Pipeline

The experimental pipeline follows a systematic multi-phase approach:

### Phase 1: Classical Baselines
Establish baseline performance using classical ML and deep learning models (LR, SVM, CNN, BiLSTM) across all three languages with TF-IDF embeddings.

### Phase 2: Template Quantum Models
Evaluate PennyLane template circuits (`StronglyEntanglingLayers`, `BasicEntanglerLayers`, `EfficientSU2`) as quantum layers in hybrid models, comparing against classical baselines.

### Phase 3: Custom VQC Models
Design and evaluate three custom VQC architectures:
- **v1**: Basic rotation + linear entanglement layers
- **v2**: Data re-uploading circuit (classical data re-encoded at each layer)
- **v3**: Ring entanglement topology

### Phase 4: Ablation Studies
Systematic ablation across:
- Embedding methods (TF-IDF, FastText, Word2Vec, BERT, mBERT, IndicBERT)
- Qubit counts (4, 6, 8)
- Circuit depths (1–6 layers)
- Activation functions and optimizers
- Training data fractions (low-resource evaluation)

### Phase 5: Low-Resource Analysis
Evaluate all model families at 1%, 5%, 10%, 25%, and 50% training data to assess sample efficiency — the core hypothesis of this research.

---

## 📊 Datasets

| Dataset | Language | Source | Emotions | Size |
|---------|----------|--------|----------|------|
| GoEmotions | English | Reddit comments | 27 emotions (mapped to 6) | ~58K |
| dair-ai/emotion | English | Twitter | 6 emotions | ~20K |
| EmoInHindi | Hindi | Native collection | 6 emotions | ~5K |
| Hindi GoEmotions | Hindi | Translated from GoEmotions | 6 emotions | ~58K |
| EmoNoBa | Bengali | Native collection | 6 emotions | ~5K |

> **Note:** Low-resource experiments use fractional subsets (1%, 5%, 10%, 25%, 50%) of the training data to evaluate sample efficiency.

---

## 🧠 Models

### Classical Models

| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression (LR) | Traditional ML | Baseline linear classifier |
| Support Vector Machine (SVM) | Traditional ML | Kernel-based classifier |
| CNN | Deep Learning | 1D convolutional text classifier |
| BiLSTM | Deep Learning | Bidirectional LSTM with attention |

### Template Quantum Models (PennyLane)

| Template | Description |
|----------|-------------|
| StronglyEntanglingLayers | Full rotation + CNOT entanglement |
| BasicEntanglerLayers | Single rotation + cyclic CNOT |
| EfficientSU2 | SU(2) rotations + linear entanglement |

### Custom VQC Models

| Version | Architecture | Key Feature |
|---------|-------------|-------------|
| v1 | Basic VQC | Simple rotation + entanglement layers |
| v2 | Data Re-uploading VQC | Classical data re-encoded at each layer |
| v3 | Ring Entanglement VQC | Circular entanglement topology |

---

## 📈 Current Project Status

- [x] Project architecture & configuration system
- [x] Multilingual preprocessing pipeline (English, Hindi, Bengali)
- [x] TF-IDF and BERT embedding implementations
- [x] Classical baselines (LR, SVM, CNN, BiLSTM)
- [x] Template quantum hybrid models (PennyLane)
- [x] Custom VQC architectures (v1, v2, v3)
- [x] Experiment runner with CLI interface
- [x] Evaluation metrics & result logging
- [x] Ablation study notebooks (embedding, VQC, activation, optimizer, low-resource)
- [ ] Data augmentation integration (back-translation, synonym, contextual)
- [ ] Memory-augmented model variants
- [ ] Final comparative analysis & paper writing

---

## 🔮 Future Work

- **Data Augmentation**: Integrate back-translation, synonym replacement, and contextual augmentation for low-resource languages
- **Memory-Augmented Models**: Implement memory bank-based classification to leverage stored representations
- **Noise Simulation**: Evaluate quantum models under realistic quantum hardware noise
- **Larger Qubit Counts**: Scale experiments to 10+ qubits on cloud quantum hardware
- **Cross-lingual Transfer**: Investigate zero-shot transfer between languages using shared quantum representations
- **Hardware Execution**: Deploy circuits on IBM Quantum / Amazon Braket for real hardware benchmarking

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{hybrid_quantum_emotion_2026,
  title     = {Hybrid Quantum-Classical Framework for Multilingual Emotion Detection},
  author    = {Soham Santra},
  year      = {2026},
  note      = {B.Tech Major Project},
  url       = {https://github.com/Soham-Santra/Hybrid-Quantum-Classical-NLP}
}
```

> **Note:** A formal citation will be updated once the associated research paper is published.

---

## 👤 Author

**Soham Santra**
- **Program:** B.Tech Major Project
- **Year:** 2026

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Built for research in quantum-enhanced natural language processing.</em><br/>
  <strong>⚛️ Where quantum computing meets multilingual NLP.</strong>
</p>
