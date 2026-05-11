"""Central configuration: paths, class list, parameter loading."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
VALIDATED_DIR: Path = DATA_DIR / "validated"
SPLITS_DIR: Path = DATA_DIR / "splits"
REJECTED_DIR: Path = DATA_DIR / "rejected"
METADATA_DIR: Path = DATA_DIR / "metadata"

MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

PARAMS_FILE: Path = PROJECT_ROOT / "params.yaml"


CLASS_NAMES: list[str] = [
    "ahmed_sharif_chaudhry",
    "ahsan_iqbal",
    "asif_ali_zardari",
    "benazir_bhutto",
    "bilawal_bhutto_zardari",
    "hamza_shehbaz",
    "imran_khan",
    "maryam_nawaz_sharif",
    "murad_ali_shah",
    "nawaz_sharif",
    "shehbaz_sharif",
    "yousaf_raza_gillani",
]

NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}


DISPLAY_NAMES: dict[str, str] = {
    "imran_khan": "Imran Khan",
    "nawaz_sharif": "Nawaz Sharif",
    "shehbaz_sharif": "Shehbaz Sharif",
    "maryam_nawaz_sharif": "Maryam Nawaz Sharif",
    "hamza_shehbaz": "Hamza Shehbaz",
    "asif_ali_zardari": "Asif Ali Zardari",
    "bilawal_bhutto_zardari": "Bilawal Bhutto Zardari",
    "benazir_bhutto": "Benazir Bhutto",
    "yousaf_raza_gillani": "Yousaf Raza Gillani",
    "murad_ali_shah": "Murad Ali Shah",
    "shah_mehmood_qureshi": "Shah Mehmood Qureshi",
    "asad_umar": "Asad Umar",
    "sheikh_rashid_ahmed": "Sheikh Rashid Ahmed",
    "fawad_chaudhry": "Fawad Chaudhry",
    "pervez_khattak": "Pervez Khattak",
    "ali_amin_gandapur": "Ali Amin Gandapur",
    "fazl_ur_rehman": "Fazl-ur-Rehman",
    "sirajul_haq": "Siraj-ul-Haq",
    "mahmood_khan_achakzai": "Mahmood Khan Achakzai",
    "akhtar_mengal": "Akhtar Mengal",
    "pervez_musharraf": "Pervez Musharraf",
    "pervez_elahi": "Pervez Elahi",
    "chaudhry_shujaat_hussain": "Chaudhry Shujaat Hussain",
    "khawaja_asif": "Khawaja Asif",
    "ahsan_iqbal": "Ahsan Iqbal",
    "hina_rabbani_khar": "Hina Rabbani Khar",
    "sherry_rehman": "Sherry Rehman",
    "asim_munir": "Asim Munir",
    "qamar_javed_bajwa": "Qamar Javed Bajwa",
    "ahmed_sharif_chaudhry": "Ahmed Sharif Chaudhry",
}


MILITARY_CLASSES: set[str] = {
    "asim_munir",
    "qamar_javed_bajwa",
    "ahmed_sharif_chaudhry",
    "pervez_musharraf",
}


_params_cache: dict[str, Any] | None = None


def load_params() -> dict[str, Any]:
    """Load params.yaml once and cache."""
    global _params_cache
    if _params_cache is None:
        with open(PARAMS_FILE) as f:
            _params_cache = yaml.safe_load(f)
    return _params_cache


def get_param(section: str, key: str, default: Any = None) -> Any:
    """Read nested value from params.yaml."""
    params = load_params()
    return params.get(section, {}).get(key, default)


def get_display_name(class_name: str) -> str:
    """Pretty name for UI/API responses."""
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def ensure_dirs() -> None:
    """Create all working directories."""
    for d in [DATA_DIR, RAW_DIR, VALIDATED_DIR, SPLITS_DIR, REJECTED_DIR,
              METADATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = "pak-public-figures-classifier"


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"First 5 classes: {CLASS_NAMES[:5]}")
    print(f"Params keys: {list(load_params().keys())}")
