"""
FairLoop Data Agent
Ingests raw data, validates schema, chunks into batches, detects demographic columns.
"""
import pandas as pd
import numpy as np
import uuid
from typing import Dict, List, Optional, Iterator
from datasets import load_dataset


class DataAgent:
    """
    Ingests raw data from HuggingFace datasets, CSV files, or DataFrames.
    Chunks into batches with metadata for downstream processing.
    """

    def __init__(self, config=None):
        from core.config import PipelineConfig
        self.config = config or PipelineConfig()
        self.raw_data: Optional[pd.DataFrame] = None
        self.batches: List[Dict] = []
        self.current_batch_idx: int = 0

    def load_dataset(
        self,
        source: str = None,
        config_name: str = None,
        split: str = None,
    ) -> pd.DataFrame:
        """Load dataset from HuggingFace Hub or CSV file."""
        source = source or self.config.dataset_name
        split = split or self.config.dataset_split

        if source.endswith('.csv'):
            self.raw_data = pd.read_csv(source)
        else:
            ds = load_dataset(source, config_name, split=split)
            self.raw_data = ds.to_pandas()

        # Clean string columns (strip whitespace)
        for col in self.raw_data.select_dtypes(include=['object']).columns:
            self.raw_data[col] = self.raw_data[col].astype(str).str.strip()

        # Validate required columns exist
        self._validate_schema()

        print(f"[DataAgent] Loaded {len(self.raw_data)} rows from {source}")
        print(f"[DataAgent] Columns: {list(self.raw_data.columns)}")
        print(f"[DataAgent] Target: {self.config.target_column}, Protected: {self.config.protected_attributes}")

        return self.raw_data

    def _validate_schema(self):
        """Validate that required columns are present."""
        missing = []
        if self.config.target_column not in self.raw_data.columns:
            missing.append(self.config.target_column)
        for attr in self.config.protected_attributes:
            if attr not in self.raw_data.columns:
                missing.append(attr)

        if missing:
            raise ValueError(
                f"[DataAgent] Missing required columns: {missing}. "
                f"Available: {list(self.raw_data.columns)}"
            )

    def detect_demographic_columns(self) -> List[str]:
        """Auto-detect columns that likely represent demographic attributes."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_dataset() first.")

        demographic_keywords = [
            'sex', 'gender', 'race', 'ethnicity', 'age', 'religion',
            'nationality', 'country', 'marital', 'disability',
        ]

        detected = []
        for col in self.raw_data.columns:
            col_lower = col.lower()
            for keyword in demographic_keywords:
                if keyword in col_lower:
                    detected.append(col)
                    break

        return detected

    def chunk_into_batches(self, batch_size: int = None) -> List[Dict]:
        """Split data into batches with metadata."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_dataset() first.")

        batch_size = batch_size or self.config.batch_size

        # Shuffle data before batching
        shuffled = self.raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

        self.batches = []
        for i in range(0, len(shuffled), batch_size):
            batch_df = shuffled.iloc[i:i + batch_size].copy()
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"

            # Compute demographic stats for this batch
            demo_stats = {}
            for attr in self.config.protected_attributes:
                if attr in batch_df.columns:
                    demo_stats[attr] = batch_df[attr].value_counts().to_dict()

            self.batches.append({
                "batch_id": batch_id,
                "data": batch_df,
                "sample_count": len(batch_df),
                "demographic_stats": demo_stats,
                "source": self.config.dataset_name,
                "is_synthetic": False,
            })

        self.current_batch_idx = 0
        print(f"[DataAgent] Created {len(self.batches)} batches of size ≤{batch_size}")
        return self.batches

    def get_next_batch(self) -> Optional[Dict]:
        """Get the next batch in sequence."""
        if self.current_batch_idx >= len(self.batches):
            return None
        batch = self.batches[self.current_batch_idx]
        self.current_batch_idx += 1
        return batch

    def get_batch_by_id(self, batch_id: str) -> Optional[Dict]:
        """Retrieve a specific batch by ID."""
        for batch in self.batches:
            if batch["batch_id"] == batch_id:
                return batch
        return None

    def replace_batch(self, batch_id: str, new_data: pd.DataFrame) -> Dict:
        """Replace a batch's data (used after remediation)."""
        for i, batch in enumerate(self.batches):
            if batch["batch_id"] == batch_id:
                new_batch_id = f"{batch_id}_r{i}"
                self.batches[i]["data"] = new_data
                self.batches[i]["sample_count"] = len(new_data)
                self.batches[i]["batch_id"] = new_batch_id
                return self.batches[i]
        raise ValueError(f"Batch {batch_id} not found")

    def add_synthetic_batch(self, synthetic_df: pd.DataFrame) -> Dict:
        """Add a new batch of synthetic data to the queue."""
        batch_id = f"batch_synth_{uuid.uuid4().hex[:8]}"
        batch = {
            "batch_id": batch_id,
            "data": synthetic_df,
            "sample_count": len(synthetic_df),
            "demographic_stats": {},
            "source": "synthetic",
            "is_synthetic": True,
        }
        self.batches.append(batch)
        return batch
