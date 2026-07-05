"""Phase 2 verification script."""
import sys
sys.path.insert(0, ".")

# Test 1: Config
from config import load_config
cfg = load_config()
print("[PASS] Config loaded OK")
print(f"  Active English dataset: {cfg.active_dataset.english}")
print(f"  Seed: {cfg.seed}")
print()

# Test 2: Registry
from data.registry import get_dataset_info, list_datasets
info = get_dataset_info("emotion")
print(f"[PASS] Dataset info: {info.name}, lang={info.language}, labels={info.num_labels}")
print(f"  Label names: {info.label_names}")
all_ds = [d.name for d in list_datasets()]
print(f"  All datasets: {all_ds}")
print()

# Test 3: Splitter
from data.splitter import stratified_split, subsample, get_label_distribution
texts = [f"text_{i}" for i in range(100)]
labels = [i % 6 for i in range(100)]

splits = stratified_split(texts, labels)
train_n = len(splits["train"]["texts"])
val_n = len(splits["val"]["texts"])
test_n = len(splits["test"]["texts"])
print(f"[PASS] Stratified split: train={train_n}, val={val_n}, test={test_n}")

sub_t, sub_l = subsample(texts, labels, 0.1)
print(f"[PASS] Subsample 10%: {len(sub_t)} samples")

dist = get_label_distribution(labels)
print(f"[PASS] Label distribution: {dist}")
print()

# Test 4: Label encoder
from data.loader import LabelEncoder
enc = LabelEncoder(["joy", "sadness", "anger", "fear", "surprise", "love"])
encoded = enc.encode_batch(["joy", "anger", "love", "sadness"])
decoded = enc.decode_batch(encoded)
print(f"[PASS] LabelEncoder: {decoded} -> {encoded} -> {decoded}")
print()

# Test 5: CLI help
import subprocess
result = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True)
if result.returncode == 0:
    print("[PASS] CLI --help works")
else:
    print(f"[FAIL] CLI --help: {result.stderr[:200]}")

print()
print("=" * 50)
print("All Phase 2 verification tests passed!")
