"""
Data utilities for protein sequence processing and embedding generation.
"""
import os
import torch
import numpy as np
from typing import List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from datasets import Dataset as HFDataset
import logging

logger = logging.getLogger(__name__)


class ProteinEmbeddingDataset(Dataset):
    """
    A Dataset that:
     - reads sequences from a FASTA
     - tokenizes them for either ESM or ProtTrans
     - runs the frozen model to get per-residue embeddings
     - aggregates them into a fixed-size vector via 'mean', 'cls', or 'max'
    """
    
    def __init__(
        self,
        fasta_path: str,
        model_name: str = "esm2_t33_650M_UR50D",
        aggregation: str = "mean",
        max_len: int = 512,
        device: str = "cpu",
    ):
        self.fasta_path = fasta_path
        self.aggregation = aggregation
        self.max_len = max_len
        self.device = device
        
        # Load sequences
        self.seqs = self._load_sequences(fasta_path)
        logger.info(f"Loaded {len(self.seqs)} sequences from {fasta_path}")
        
        # Initialize model
        self._init_model(model_name)
        
    def _load_sequences(self, fasta_path: str) -> List[str]:
        """Load sequences from FASTA file."""
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
            
        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
        
        if not sequences:
            raise ValueError(f"No sequences found in {fasta_path}")
            
        return sequences
    
    def _init_model(self, model_name: str):
        """Initialize the protein language model."""
        if model_name.startswith("esm"):
            self._init_esm_model(model_name)
        else:
            self._init_prottrans_model(model_name)
    
    def _init_esm_model(self, model_name: str):
        """Initialize ESM model."""
        import esm
        
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)
        self.backend = "esm"
        
        logger.info(f"Initialized ESM model: {model_name}")
    
    def _init_prottrans_model(self, model_name: str):
        """Initialize ProtTrans model."""
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval().to(self.device)
        self.backend = "prottrans"
        
        logger.info(f"Initialized ProtTrans model: {model_name}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][: self.max_len]  # truncate if too long
        
        if self.backend == "esm":
            return self._get_esm_embedding(seq, idx)
        else:
            return self._get_prottrans_embedding(seq)
    
    def _get_esm_embedding(self, seq: str, idx: int) -> torch.Tensor:
        """Get embedding from ESM model."""
        batch = [(str(idx), seq)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.model.num_layers])
        
        reps = out["representations"][self.model.num_layers]  # (1, L, C)
        mask = (tokens != self.alphabet.padding_idx).unsqueeze(-1)  # (1, L, 1)
        
        return self._aggregate_embeddings(reps, mask)
    
    def _get_prottrans_embedding(self, seq: str) -> torch.Tensor:
        """Get embedding from ProtTrans model."""
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=False)
        
        reps = out.last_hidden_state  # (1, L, C)
        mask = enc["attention_mask"].unsqueeze(-1).bool()  # (1, L, 1)
        
        return self._aggregate_embeddings(reps, mask)
    
    def _aggregate_embeddings(
        self, reps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate per-residue embeddings into fixed-size vector."""
        reps = reps.squeeze(0)  # (L, C)
        mask = mask.squeeze(0)  # (L, 1)

        if self.aggregation == "mean":
            summed = (reps * mask).sum(0)  # (C,)
            lengths = mask.sum(0).clamp(min=1)  # (C,) broadcastable
            emb = summed / lengths
        elif self.aggregation == "max":
            reps_masked = reps.masked_fill(~mask, -1e9)
            emb, _ = reps_masked.max(0)  # (C,)
        elif self.aggregation == "cls":
            emb = reps[0]  # First token
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return emb.cpu()


class EmbeddingWrapper(Dataset):
    """Wrapper to format embeddings for Lightning modules."""
    
    def __init__(self, embed_ds):
        self.embed_ds = embed_ds

    def __len__(self):
        return len(self.embed_ds)

    def __getitem__(self, idx):
        emb = self.embed_ds[idx]
        return {"emb": emb}


def create_sample_fasta(fasta_path: str, num_samples: int = 10):
    """Create a sample FASTA file for testing."""
    sample_sequences = [
        "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        "GILGYTEAQVKILDGGSGFYTNLTMATPLKAPIK",
        "MTIQTGLDSTGTTMTVVESKDLKELLEAQQGIQAYSQVGR",
        "MKATQITLILVLGLLVSLGAAVQADQNPTANIPKGAMKPT",
        "MGSSHHHHHHSSGLVPRGSHMLEEILLKKLANPVGSAYK",
        "MSSIIVGSDLTRIKEIKQAVEARKQGVNPDEVVDIGRTM",
        "MTNLYSQPQKGDYKTLLFQNVQGYDLYQKQGKVALFGSD",
        "MSTQKNQKQLVNLGNLLRQSVEQHVQRSLPGIKEFQRGA",
        "MVKQHSEPKNLQVLINQHGQSLQKQPEQGQKQHLIEVLA",
        "MSTELEHKLQNSQTILLANPSQVDQKQVNLLQNHGASLQ"
    ]
    
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sample_sequences[:num_samples]):
            f.write(f">protein_{i+1}\n{seq}\n")
    
    logger.info(f"Created sample FASTA with {num_samples} sequences at {fasta_path}")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
