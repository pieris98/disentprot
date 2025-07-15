"""
Disentanglement evaluation metrics for protein representations.
"""
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def beta_vae_metric(z: torch.Tensor, factors: torch.Tensor, num_train: int = 10000) -> float:
    """
    Compute β-VAE metric for disentanglement evaluation.
    
    Args:
        z: Latent representations (N, latent_dim)
        factors: Ground truth factors (N, num_factors)
        num_train: Number of samples for training
        
    Returns:
        β-VAE metric score
    """
    z_np = z.cpu().numpy() if torch.is_tensor(z) else z
    factors_np = factors.cpu().numpy() if torch.is_tensor(factors) else factors
    
    # Handle small datasets
    if z_np.shape[0] < 5:
        logger.warning("Dataset too small for β-VAE metric, returning default score")
        return 0.5
    
    num_factors = factors_np.shape[1]
    latent_dim = z_np.shape[1]
    
    scores = []
    
    for factor_idx in range(num_factors):
        try:
            # Check if factor has variance
            if len(np.unique(factors_np[:, factor_idx])) < 2:
                scores.append(0.5)  # Default score for constant factors
                continue
                
            # Train classifier for each factor
            train_size = min(num_train, max(3, int(0.8 * len(z_np))))  # At least 3 samples
            X_train, X_test, y_train, y_test = train_test_split(
                z_np, factors_np[:, factor_idx], 
                train_size=train_size,
                random_state=42,
                stratify=factors_np[:, factor_idx] if len(np.unique(factors_np[:, factor_idx])) > 1 else None
            )
            
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
            
        except Exception as e:
            logger.warning(f"Error computing β-VAE metric for factor {factor_idx}: {e}")
            scores.append(0.5)  # Default score on error
    
    return np.mean(scores) if scores else 0.5


def mig_score(z: torch.Tensor, factors: torch.Tensor) -> float:
    """
    Compute Mutual Information Gap (MIG) score.
    
    Args:
        z: Latent representations (N, latent_dim)
        factors: Ground truth factors (N, num_factors)
        
    Returns:
        MIG score
    """
    z_np = z.cpu().numpy() if torch.is_tensor(z) else z
    factors_np = factors.cpu().numpy() if torch.is_tensor(factors) else factors
    
    # Discretize latent variables for MI computation
    z_discrete = np.digitize(z_np, np.linspace(z_np.min(), z_np.max(), 20))
    
    num_factors = factors_np.shape[1]
    latent_dim = z_np.shape[1]
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((latent_dim, num_factors))
    
    for i in range(latent_dim):
        for j in range(num_factors):
            mi_matrix[i, j] = mutual_information(z_discrete[:, i], factors_np[:, j])
    
    # Compute MIG
    mig = 0
    for j in range(num_factors):
        mi_j = mi_matrix[:, j]
        if len(mi_j) > 1:
            # Sort in descending order
            mi_j_sorted = np.sort(mi_j)[::-1]
            mig += (mi_j_sorted[0] - mi_j_sorted[1]) / np.max(mi_j)
    
    return mig / num_factors if num_factors > 0 else 0


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two discrete variables."""
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(x, y)


def sap_score(z: torch.Tensor, factors: torch.Tensor) -> float:
    """
    Compute Separated Attribute Predictability (SAP) score.
    
    Args:
        z: Latent representations (N, latent_dim)
        factors: Ground truth factors (N, num_factors)
        
    Returns:
        SAP score
    """
    z_np = z.cpu().numpy() if torch.is_tensor(z) else z
    factors_np = factors.cpu().numpy() if torch.is_tensor(factors) else factors
    
    num_factors = factors_np.shape[1]
    latent_dim = z_np.shape[1]
    
    # Train predictors for each factor
    scores_matrix = np.zeros((num_factors, latent_dim))
    
    for factor_idx in range(num_factors):
        for latent_idx in range(latent_dim):
            # Use single latent dimension to predict factor
            X = z_np[:, latent_idx].reshape(-1, 1)
            y = factors_np[:, factor_idx]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            scores_matrix[factor_idx, latent_idx] = accuracy_score(y_test, y_pred)
    
    # Compute SAP
    sap = 0
    for factor_idx in range(num_factors):
        scores = scores_matrix[factor_idx]
        if len(scores) > 1:
            scores_sorted = np.sort(scores)[::-1]
            sap += scores_sorted[0] - scores_sorted[1]
    
    return sap / num_factors if num_factors > 0 else 0


def dci_disentanglement(z: torch.Tensor, factors: torch.Tensor) -> Dict[str, float]:
    """
    Compute DCI (Disentanglement, Completeness, Informativeness) metrics.
    
    Args:
        z: Latent representations (N, latent_dim)
        factors: Ground truth factors (N, num_factors)
        
    Returns:
        Dictionary with DCI scores
    """
    z_np = z.cpu().numpy() if torch.is_tensor(z) else z
    factors_np = factors.cpu().numpy() if torch.is_tensor(factors) else factors
    
    num_factors = factors_np.shape[1]
    latent_dim = z_np.shape[1]
    
    # Compute importance matrix using Random Forest
    importance_matrix = np.zeros((num_factors, latent_dim))
    
    for factor_idx in range(num_factors):
        # Train Random Forest to predict factor from all latents
        X_train, X_test, y_train, y_test = train_test_split(
            z_np, factors_np[:, factor_idx], test_size=0.2, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_matrix[factor_idx] = rf.feature_importances_
    
    # Normalize importance matrix
    importance_matrix = importance_matrix / (importance_matrix.sum(axis=1, keepdims=True) + 1e-12)
    
    # Compute Disentanglement
    disentanglement = 1 - scipy_entropy(importance_matrix.T, base=2, axis=0).mean()
    
    # Compute Completeness
    completeness = 1 - scipy_entropy(importance_matrix, base=2, axis=1).mean()
    
    # Compute Informativeness (how well can we predict factors)
    informativeness = 0
    for factor_idx in range(num_factors):
        X_train, X_test, y_train, y_test = train_test_split(
            z_np, factors_np[:, factor_idx], test_size=0.2, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        informativeness += accuracy_score(y_test, y_pred)
    
    informativeness /= num_factors
    
    return {
        'disentanglement': disentanglement,
        'completeness': completeness, 
        'informativeness': informativeness
    }


def scipy_entropy(pk, qk=None, base=None, axis=0):
    """Compute entropy (from scipy.stats)."""
    pk = np.asarray(pk)
    pk = pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = np.where(pk == 0, 0, pk * np.log(pk))
    else:
        qk = np.asarray(qk)
        pk, qk = np.broadcast_arrays(pk, qk)
        qk = qk / np.sum(qk, axis=axis, keepdims=True)
        vec = np.where(pk == 0, 0, pk * np.log(pk / qk))
    
    S = -np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    
    return S


class DisentanglementEvaluator:
    """Comprehensive disentanglement evaluation."""
    
    def __init__(self):
        self.metrics = {
            'beta_vae': beta_vae_metric,
            'mig': mig_score,
            'sap': sap_score,
            'dci': dci_disentanglement
        }
    
    def evaluate(
        self, 
        z: torch.Tensor, 
        factors: Optional[torch.Tensor] = None,
        synthetic_factors: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate disentanglement metrics.
        
        Args:
            z: Latent representations
            factors: Ground truth factors (if available)
            synthetic_factors: Whether to generate synthetic factors for evaluation
            
        Returns:
            Dictionary of metric scores
        """
        # Handle very small datasets
        if z.shape[0] < 3:
            logger.warning(f"Dataset too small ({z.shape[0]} samples) for evaluation")
            return {
                'beta_vae': 0.5,
                'mig': 0.0,
                'sap': 0.5,
                'dci_disentanglement': 0.5,
                'dci_completeness': 0.5,
                'dci_informativeness': 0.5,
                'note': 'Dataset too small for meaningful evaluation'
            }
        
        if factors is None and synthetic_factors:
            factors = self._generate_synthetic_factors(z)
        elif factors is None:
            logger.warning("No factors provided and synthetic_factors=False")
            return {'error': 'No factors available for evaluation'}
        
        results = {}
        
        for metric_name, metric_fn in self.metrics.items():
            try:
                if metric_name == 'dci':
                    result = metric_fn(z, factors)
                    results.update({f'dci_{k}': v for k, v in result.items()})
                else:
                    results[metric_name] = metric_fn(z, factors)
                
                logger.info(f"Computed {metric_name}: {results.get(metric_name, 'N/A')}")
                
            except Exception as e:
                logger.warning(f"Error computing {metric_name}: {e}")
                results[metric_name] = 0.5 if 'beta' in metric_name or 'sap' in metric_name else 0.0
        
        return results
    
    def _generate_synthetic_factors(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthetic factors for evaluation when ground truth unavailable."""
        # Simple approach: use clustering or PCA to create synthetic factors
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        z_np = z.cpu().numpy() if torch.is_tensor(z) else z
        n_samples = z_np.shape[0]
        
        # Handle very small datasets
        if n_samples < 3:
            logger.warning(f"Dataset too small ({n_samples} samples) for meaningful evaluation")
            # Create dummy factors for very small datasets
            factors = []
            factors.append(np.arange(n_samples))  # Simple enumeration
            factors.append(np.zeros(n_samples))   # All same class
            factors.append(np.random.randint(0, max(2, n_samples), size=n_samples))  # Random
            synthetic_factors = np.column_stack(factors)
            return torch.tensor(synthetic_factors, dtype=torch.long)
        
        # Create synthetic factors using different methods
        factors = []
        
        # Factor 1: K-means clustering (limit clusters to reasonable number)
        max_clusters = min(5, max(2, n_samples // 2))  # At least 2, at most 5 or n_samples//2
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        factors.append(kmeans.fit_predict(z_np))
        
        # Factor 2: PCA-based grouping
        try:
            pca = PCA(n_components=1)
            pca_scores = pca.fit_transform(z_np).flatten()
            # Create quantile-based groups
            n_groups = min(4, max(2, n_samples // 2))
            if n_groups > 1:
                pca_quantiles = np.quantile(pca_scores, np.linspace(0, 1, n_groups + 1)[1:-1])
                pca_factor = np.digitize(pca_scores, pca_quantiles)
            else:
                pca_factor = np.zeros(n_samples)
            factors.append(pca_factor)
        except Exception as e:
            logger.warning(f"PCA factor generation failed: {e}")
            factors.append(np.zeros(n_samples))
        
        # Factor 3: Random grouping (control)
        n_random_groups = min(3, max(2, n_samples // 2))
        random_factor = np.random.randint(0, n_random_groups, size=n_samples)
        factors.append(random_factor)
        
        synthetic_factors = np.column_stack(factors)
        return torch.tensor(synthetic_factors, dtype=torch.long)


def evaluate_protein_representations(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate protein representations using disentanglement metrics.
    
    Args:
        model: Trained model (VAE or SimCLR)
        dataloader: DataLoader for evaluation
        device: Device for computation
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    model.to(device)
    
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            emb = batch['emb'].to(device)
            
            if hasattr(model, 'encode_batch'):
                # VAE model
                z = model.encode_batch(emb)
            elif hasattr(model, 'projection_head'):
                # SimCLR model
                z = model.projection_head(emb)
            else:
                # Generic model
                z = model(emb)
            
            representations.append(z.cpu())
    
    all_representations = torch.cat(representations, dim=0)
    
    # Evaluate disentanglement
    evaluator = DisentanglementEvaluator()
    results = evaluator.evaluate(all_representations)
    
    return results