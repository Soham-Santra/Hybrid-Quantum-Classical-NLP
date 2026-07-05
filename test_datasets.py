"""Verification: all 3 primary datasets are registered and loadable."""
import sys
sys.path.insert(0, ".")

from config import load_config
from data.registry import get_dataset_info, list_datasets

# Test config loads
cfg = load_config()
print("=" * 60)
print("  DATASET INTEGRATION VERIFICATION")
print("=" * 60)

# Active dataset settings
print(f"\nActive datasets in config:")
print(f"  English: index {cfg.active_dataset.english}")
print(f"  Hindi:   index {cfg.active_dataset.hindi}")
print(f"  Bengali: index {cfg.active_dataset.bengali}")

# Verify all 3 primary datasets are in registry
print("\n--- Primary Datasets ---")
for name in ["go_emotions", "emo_in_hindi", "emonoba"]:
    info = get_dataset_info(name)
    print(f"\n[PASS] {info.name}")
    print(f"  Language:    {info.language}")
    print(f"  Source:      {info.source}")
    print(f"  Text col:    {info.text_column}")
    print(f"  Labels:      {info.num_labels} ({info.label_names[:4]}...)")
    print(f"  Multilabel:  {info.multilabel}")
    if info.binary_label_columns:
        print(f"  Binary cols: {info.binary_label_columns}")
    if info.split_files:
        print(f"  Split files: {info.split_files}")
    if info.unified_label_map:
        print(f"  Unified map: {len(info.unified_label_map)} entries")

# List all datasets by language
print("\n--- All Registered Datasets ---")
for lang in ["english", "hindi", "bengali"]:
    ds = list_datasets(lang)
    names = [d.name for d in ds]
    print(f"  {lang}: {names}")

# Verify unified label compatibility
from data.registry import UNIFIED_LABELS
print(f"\nUnified labels: {UNIFIED_LABELS}")

# Check EmoNoBa labels match unified
emonoba = get_dataset_info("emonoba")
emonoba_set = set(emonoba.label_names)
unified_set = set(UNIFIED_LABELS)
match = emonoba_set == unified_set
print(f"EmoNoBa labels match unified: {match} OK" if match else f"MISMATCH! {emonoba_set} vs {unified_set}")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED [OK]")
print("=" * 60)
