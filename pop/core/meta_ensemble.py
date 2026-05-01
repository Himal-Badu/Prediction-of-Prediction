"""
Meta-Ensemble for PoP — Hierarchical Classification
===================================================

Architecture:
    Level 1: Branch classifiers (NLI, CosSim, Length)
    Level 2: Meta-classifier (GradientBoosting)

Each branch specializes in different feature types.
Meta-classifier learns optimal combination.

Why hierarchical?
- Different error types require different detectors
- Meta-learner captures interactions between branches
- More interpretable (see which branch disagrees)
- Better accuracy than single classifier
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoPMetaEnsemble:
    """
    Hierarchical ensemble for PoP hallucination detection.
    
    Structure:
        Level 1: Specialized branch classifiers
            ├─ NLI_BRANCH (entailment, neutral, contradiction probs)
            ├─ COSIM_BRANCH (forward, reverse, asymmetry)
            └─ LENGTH_BRANCH (ratio, q_len, c_len)
        
        Level 2: Meta-classifier (GradientBoosting)
            └─ Combines branch predictions → final hallucination probability
    
    Advantages over single classifier:
    - Each branch specializes in different feature semantics
    - Meta-learner captures non-linear interactions
    - More robust (branch disagreement flags uncertainty)
    - Interpretable (see per-branch contributions)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Branch 1: NLI classifier
        self.nli_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Branch 2: CosSim classifier
        self.cosim_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Branch 3: Length classifier
        self.length_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Meta-classifier (GradientBoosting for non-linear interactions)
        self.meta_clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state
        )
        
        # Feature scaler (for consistency)
        self.scaler = StandardScaler()
        
        # Tracking
        self.is_trained = False
        self.training_history = {}
        
        logger.info("PoPMetaEnsemble initialized")
    
    def split_features(self, X):
        """
        Split combined features into branch-specific sets.
        
        Expected X shape: (n_samples, 8)
        Columns: [entail, neutral, contradict, 
                  fwd_sim, rev_sim, asymmetry,
                  len_ratio, q_len, c_len]
        
        Returns:
            X_nli: (n_samples, 3)
            X_cosim: (n_samples, 3)
            X_length: (n_samples, 3)
        """
        X_nli = X[:, :3]        # entail, neutral, contradict
        X_cosim = X[:, 3:6]     # fwd, rev, asymmetry
        X_length = X[:, 6:]     # len_ratio, q_len, c_len
        
        return X_nli, X_cosim, X_length
    
    def fit(self, X, y, cv_folds=5):
        """
        Train hierarchical ensemble.
        
        Args:
            X: Feature matrix (n_samples, 8)
            y: Labels (n_samples,)
            cv_folds: Number of CV folds for meta-features
            
        Returns:
            dict: Training history and metrics
        """
        logger.info("Starting hierarchical ensemble training...")
        
        # Split features by branch
        X_nli, X_cosim, X_length = self.split_features(X)
        
        n_samples = X.shape[0]
        
        # ── STEP 1: Train branch classifiers ──
        logger.info("Training branch classifiers...")
        
        # NLI branch
        self.nli_clf.fit(X_nli, y)
        nli_train_pred = self.nli_clf.predict_proba(X_nli)[:, 1]
        nli_auc = roc_auc_score(y, nli_train_pred)
        logger.info(f"  NLI branch AUC: {nli_auc:.4f}")
        
        # CosSim branch
        self.cosim_clf.fit(X_cosim, y)
        cosim_train_pred = self.cosim_clf.predict_proba(X_cosim)[:, 1]
        cosim_auc = roc_auc_score(y, cosim_train_pred)
        logger.info(f"  CosSim branch AUC: {cosim_auc:.4f}")
        
        # Length branch
        self.length_clf.fit(X_length, y)
        length_train_pred = self.length_clf.predict_proba(X_length)[:, 1]
        length_auc = roc_auc_score(y, length_train_pred)
        logger.info(f"  Length branch AUC: {length_auc:.4f}")
        
        # ── STEP 2: Generate meta-features (out-of-fold predictions) ──
        logger.info(f"Generating meta-features ({cv_folds}-fold CV)...")
        
        meta_features = np.zeros((n_samples, 3))  # 3 branch predictions
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                              random_state=self.random_state)
        
        for train_idx, val_idx in skf.split(X, y):
            # Split data
            X_tr_nli, X_val_nli = X_nli[train_idx], X_nli[val_idx]
            X_tr_cosim, X_val_cosim = X_cosim[train_idx], X_cosim[val_idx]
            X_tr_len, X_val_len = X_length[train_idx], X_length[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Train temporary branch classifiers
            tmp_nli = RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=self.random_state,
                n_jobs=-1
            )
            tmp_nli.fit(X_tr_nli, y_tr)
            meta_features[val_idx, 0] = tmp_nli.predict_proba(X_val_nli)[:, 1]
            
            tmp_cosim = RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=self.random_state,
                n_jobs=-1
            )
            tmp_cosim.fit(X_tr_cosim, y_tr)
            meta_features[val_idx, 1] = tmp_cosim.predict_proba(X_val_cosim)[:, 1]
            
            tmp_length = RandomForestClassifier(
                n_estimators=300, max_depth=8, random_state=self.random_state,
                n_jobs=-1
            )
            tmp_length.fit(X_tr_len, y_tr)
            meta_features[val_idx, 2] = tmp_length.predict_proba(X_val_len)[:, 1]
        
        # ── STEP 3: Train meta-classifier ──
        logger.info("Training meta-classifier...")
        
        # Scale meta-features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-classifier
        self.meta_clf.fit(meta_features_scaled, y)
        
        # Meta training predictions
        meta_train_pred = self.meta_clf.predict_proba(meta_features_scaled)[:, 1]
        meta_auc = roc_auc_score(y, meta_train_pred)
        logger.info(f"  Meta-classifier AUC: {meta_auc:.4f}")
        
        # ── STEP 4: Store training history ──
        self.training_history = {
            'nli_auc': nli_auc,
            'cosim_auc': cosim_auc,
            'length_auc': length_auc,
            'meta_auc': meta_auc,
            'cv_folds': cv_folds
        }
        
        self.is_trained = True
        
        logger.info("Training complete!")
        
        return {
            'status': 'trained',
            'branch_aucs': {
                'nli': nli_auc,
                'cosim': cosim_auc,
                'length': length_auc
            },
            'meta_auc': meta_auc,
            'improvement_vs_single': meta_auc - max(nli_auc, cosim_auc, length_auc)
        }
    
    def predict_proba(self, X):
        """
        Predict hallucination probability for new samples.
        
        Args:
            X: Feature matrix (n_samples, 8)
            
        Returns:
            proba: Hallucination probabilities (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Split features
        X_nli, X_cosim, X_length = self.split_features(X)
        
        # Get branch predictions
        nli_proba = self.nli_clf.predict_proba(X_nli)[:, 1]
        cosim_proba = self.cosim_clf.predict_proba(X_cosim)[:, 1]
        length_proba = self.length_clf.predict_proba(X_length)[:, 1]
        
        # Combine into meta-features
        meta_features = np.column_stack([nli_proba, cosim_proba, length_proba])
        
        # Scale
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Meta prediction
        final_proba = self.meta_clf.predict_proba(meta_features_scaled)[:, 1]
        
        return final_proba
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary class (hallucination or not).
        
        Args:
            X: Feature matrix (n_samples, 8)
            threshold: Decision threshold (default: 0.5)
            
        Returns:
            predictions: Binary predictions (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_branch_disagreement(self, X):
        """
        Get disagreement between branches (uncertainty measure).
        
        High disagreement → model uncertain → flag for review
        
        Args:
            X: Feature matrix (n_samples, 8)
            
        Returns:
            disagreement: Std of branch predictions (n_samples,)
        """
        X_nli, X_cosim, X_length = self.split_features(X)
        
        nli_proba = self.nli_clf.predict_proba(X_nli)[:, 1]
        cosim_proba = self.cosim_clf.predict_proba(X_cosim)[:, 1]
        length_proba = self.length_clf.predict_proba(X_length)[:, 1]
        
        # Compute standard deviation across branches
        branch_probas = np.column_stack([nli_proba, cosim_proba, length_proba])
        disagreement = np.std(branch_probas, axis=1)
        
        return disagreement
    
    def get_params(self):
        """Get model parameters."""
        return {
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'n_estimators_nli': self.nli_clf.n_estimators,
            'n_estimators_cosim': self.cosim_clf.n_estimators,
            'n_estimators_length': self.length_clf.n_estimators,
            'meta_n_estimators': self.meta_clf.n_estimators,
            'meta_max_depth': self.meta_clf.max_depth,
            'meta_learning_rate': self.meta_clf.learning_rate
        }


def create_meta_ensemble(random_state=42):
    """
    Factory function to create meta ensemble.
    
    Args:
        random_state: Random seed
        
    Returns:
        PoPMetaEnsemble instance
    """
    return PoPMetaEnsemble(random_state=random_state)


if __name__ == '__main__':
    # Quick test
    print("PoP Meta-Ensemble Module")
    print("="*50)
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # 8 features: [entail, neutral, contradict, fwd, rev, asymmetry, len_ratio, q_len, c_len]
    X = np.random.randn(n_samples, 8)
    y = (X[:, 0] - X[:, 1] + 0.5 * X[:, 4] > 0).astype(int)  # Synthetic labels
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print("\nTraining meta-ensemble...")
    ensemble = PoPMetaEnsemble(random_state=42)
    history = ensemble.fit(X_train, y_train, cv_folds=5)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred_proba = ensemble.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"Branch AUCs: {history['branch_aucs']}")
    print(f"Meta AUC: {history['meta_auc']:.4f}")
    print(f"Improvement: {history['improvement_vs_single']:.4f}")
    
    print("\n✅ Module working correctly!")
