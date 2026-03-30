from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


SUPPORTED_METHODS = {"fullindexing", "key-blocking", "token-blocking", "minhash-lsh"}
METHOD_ALIASES = {
    "key": "key-blocking",
    "token": "token-blocking",
    "minhash": "minhash-lsh",
}

_TOKEN_RE = re.compile(r"\w+")


def _cg_block(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return config.get("candidate_generation", {}) or {}


def _as_field_list(value: Any, field_name: str, required: bool = True) -> List[str]:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return []
    if isinstance(value, str):
        fields = [value]
    elif isinstance(value, (list, tuple, set)):
        fields = [str(v) for v in value]
    else:
        raise ValueError(f"{field_name} must be a string or list of strings.")
    fields = [f.strip() for f in fields if str(f).strip()]
    if required and not fields:
        raise ValueError(f"{field_name} must contain at least one field.")
    return fields


def _ensure_rid(df: pd.DataFrame) -> None:
    if "rid" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'rid' column.")


def _rid_to_str(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Invalid rid value: {value!r}")
    if value is None:
        raise ValueError("rid cannot be None")
    rid = str(value).strip()
    if not rid:
        raise ValueError("rid cannot be empty")
    return rid


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _record_text(row: pd.Series, fields: Sequence[str]) -> str:
    parts: List[str] = []
    for field in fields:
        if field not in row.index:
            continue
        text = _safe_text(row[field])
        if text:
            parts.append(text)
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _stable_int_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _char_shingles(text: str, shingle_size: int) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return []
    if shingle_size <= 1:
        return list(normalized)
    if len(normalized) < shingle_size:
        return [normalized]
    return [normalized[i : i + shingle_size] for i in range(len(normalized) - shingle_size + 1)]


def _compute_key_blocking_features(df: pd.DataFrame, config: Optional[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    keys = _as_field_list(
        _cg_block(config).get("key_blocking", {}).get("keys"),
        "candidate_generation.key_blocking.keys",
        required=True,
    )
    features: List[Tuple[str, List[str]]] = []
    for _, row in df.iterrows():
        rid = _rid_to_str(row["rid"])
        emitted: List[str] = []
        composite_parts: List[str] = []
        for field in keys:
            value = _safe_text(row[field]) if field in row.index else ""
            composite_parts.append(value.lower())
            if value:
                emitted.append(f"{field}:{value.lower()}")
        composite = "|".join(composite_parts).strip("|")
        if composite:
            emitted.append(f"__all__:{composite}")
        features.append((rid, list(dict.fromkeys(emitted))))
    return features


def _compute_token_blocking_features(df: pd.DataFrame, config: Optional[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    fields = _as_field_list(
        _cg_block(config).get("token_blocking", {}).get("field"),
        "candidate_generation.token_blocking.field",
        required=True,
    )
    features: List[Tuple[str, List[str]]] = []
    for _, row in df.iterrows():
        rid = _rid_to_str(row["rid"])
        text = _record_text(row, fields)
        tokens = list(dict.fromkeys(_tokenize(text)))
        features.append((rid, tokens))
    return features


def _compute_minhash_lsh_features(df: pd.DataFrame, config: Optional[Dict[str, Any]]) -> List[Tuple[str, List[int]]]:
    cg_cfg = _cg_block(config)
    mh_cfg = cg_cfg.get("minhash_lsh", {}) if isinstance(cg_cfg.get("minhash_lsh", {}), dict) else {}
    fields = _as_field_list(mh_cfg.get("field"), "candidate_generation.minhash_lsh.field", required=True)

    num_perm = int(mh_cfg.get("num_perm", 128))
    bands = int(mh_cfg.get("bands", 32))
    rows_per_band = int(mh_cfg.get("rows_per_band", 4))
    shingle_size = int(mh_cfg.get("shingle_size", 3))
    seed = int(mh_cfg.get("seed", cg_cfg.get("random_seed", 42) or 42))

    if num_perm <= 0 or bands <= 0 or rows_per_band <= 0 or shingle_size <= 0:
        raise ValueError("minhash_lsh parameters num_perm/bands/rows_per_band/shingle_size must be positive.")

    expected_sig_len = bands * rows_per_band
    sig_len = num_perm
    if sig_len < rows_per_band:
        rows_per_band = sig_len
        bands = 1
    elif expected_sig_len != sig_len:
        usable_bands = sig_len // rows_per_band
        if usable_bands <= 0:
            usable_bands = 1
            rows_per_band = sig_len
        bands = usable_bands

    features: List[Tuple[str, List[int]]] = []
    for _, row in df.iterrows():
        rid = _rid_to_str(row["rid"])
        text = _record_text(row, fields)
        shingles = _char_shingles(text, shingle_size)
        if not shingles:
            features.append((rid, []))
            continue

        signature: List[int] = []
        for perm_idx in range(sig_len):
            min_val: Optional[int] = None
            salt = f"{seed}:{perm_idx}:"
            for sh in shingles:
                h = _stable_int_hash(salt + sh)
                if min_val is None or h < min_val:
                    min_val = h
            signature.append(min_val if min_val is not None else 0)

        bucket_ids: List[int] = []
        for band_idx in range(bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            if end > len(signature):
                break
            band_sig = signature[start:end]
            bucket = _stable_int_hash(f"{seed}:band:{band_idx}:" + ",".join(str(v) for v in band_sig))
            bucket_ids.append(bucket)

        features.append((rid, bucket_ids))
    return features


def compute_features(df: pd.DataFrame, method: str, config: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
    if df is None:
        return []
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame.")

    _ensure_rid(df)
    if df.empty:
        return []

    if method == "fullindexing":
        return [(_rid_to_str(row["rid"]), None) for _, row in df.iterrows()]
    if method == "key-blocking":
        return _compute_key_blocking_features(df, config)
    if method == "token-blocking":
        return _compute_token_blocking_features(df, config)
    if method == "minhash-lsh":
        return _compute_minhash_lsh_features(df, config)

    raise ValueError(f"Unsupported method: {method}")


__all__ = [
    "compute_features",
]
