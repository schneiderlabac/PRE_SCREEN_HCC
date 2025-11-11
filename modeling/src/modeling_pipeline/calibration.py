"""
Enhanced Calibration layer for machine learning pipeline.

Includes clinical visualization and Excel export functionality.
Uses proper logging for internal operations while keeping user-facing print output.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    roc_curve, auc, average_precision_score,
    brier_score_loss, log_loss, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime



plt.style.use('default')
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']



# Configure module logger
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="findfont: Font family.*not found")


class CalibrationLayer:
    """
    Enhanced calibration layer with clinical visualization and Excel export.

    New features:
    - Clinical calibration assessment visualization
    - Excel export of calibration metrics
    - Comprehensive tracking across multiple evaluations
    - Proper logging for debugging and production deployment

    # A key question is on which data the calibration is performed. Below, there are two functions:
    - A) Calibration on hold-out/validation set (20% of original data). When doing this, the calibration needs to be evaluated on an external dataset. Moreover, this limits the number of samples available for calibraiton
    - B) Calibration on the training set using cross-validation (e.g. 5-fold). This allows to use all data for calibration, but may lead to overfitting. The evaluation can then be done on the hold-out set or an external dataset.

    """

    def __init__(self, log_level=logging.DEBUG, use_balanced_calibration=True):
        """
        Initialize calibration layer.

        Parameters:
        -----------
        log_level : int
            Logging level (default: logging.INFO)
        use_balanced_calibration : bool, default=True
            If True, use balanced class weights when calculating calibration metrics.
            Recommended for imbalanced datasets where the positive class is overrepresented.
        """

        self.calibrated_models = []
        self.calibrated_master = None
        self.is_calibrated = False
        self.calibrated_on = None
        self.calibration_method = None
        self.calibration_cv = 5
        self.calibration_metrics = {}
        self.evaluation_history = []
        self.use_balanced_calibration = use_balanced_calibration  # NEW: store preference

        # Default plotting colors (can be overridden per-call by passing `colors` dict)
        self.default_plot_colors = {
            'color_prior': "#C13617",
            'edgecolor_prior': 'darkred',
            'color_past': '#385579',
            'edgecolor_past': 'darkblue',
            'control_fill': '#E8E8E8',
            'control_edge': "#A1A1A1",
            'cal_marker_color': '#4ECDC4',
            'cal_marker_edge': '#006B66',
            'before_bar': 'lightcoral',
            'after_bar': 'lightgreen'
        }

        # Set up instance logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        self.logger.info("Enhanced Calibration layer initialized")
        if use_balanced_calibration:
            self.logger.info("Using balanced class weights for calibration metrics")

    def _get_pipeline(self):
        """Get the pipeline by inspecting the call stack."""
        import inspect

        # First try weak reference if it exists
        if hasattr(self, '_pipeline_ref'):
            pipeline = self._pipeline_ref()
            if pipeline is not None:
                return pipeline

        # Fallback: Look at the call stack
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the caller
            caller_frame = frame.f_back.f_back  # Skip one level to get to the actual caller

            # Look for variables that contain this calibration
            for name, obj in caller_frame.f_locals.items():
                if (hasattr(obj, 'calibration') and
                    obj.calibration is self and
                    hasattr(obj, 'name')):
                    return obj

            # Also check globals
            for name, obj in caller_frame.f_globals.items():
                if (hasattr(obj, 'calibration') and
                    obj.calibration is self and
                    hasattr(obj, 'name')):
                    return obj

        finally:
            del frame

        raise ValueError("Cannot find pipeline containing this calibration object")

    def calibrate_with_cv_on_train(self, method='sigmoid', n_folds=5, save: bool = True, save_compress=('zlib', 3)):
        """
        Calibrate using cross-validation on the TRAINING set (80%).

        This is the recommended approach when you have limited events in your holdout set.
        It uses CV on the training data for calibration, then you can evaluate on the
        holdout 20% which remains completely independent.

        Parameters:
        -----------
        method : str, default='sigmoid'
            Calibration method: 'sigmoid' (Platt scaling) or 'isotonic'
        n_folds : int, default=5
            Number of cross-validation folds to use on training data

        Best Practice:
        --------------
        - Calibrate on 80% training set using CV (this method)
        - Evaluate calibration on 20% holdout set (evaluate_on_holdout)
        - Final external validation on completely external dataset
        """
        self.logger.info("="*80)
        self.logger.info("Starting CV calibration on TRAINING set")
        self.logger.info(f"Method: {method}, CV folds: {n_folds}")

        print(f"\n{'='*80}")
        print("CALIBRATING WITH CROSS-VALIDATION ON TRAINING SET")
        print(f"{'='*80}")
        print("Strategy: Use internal CV on 80% training data for calibration")
        print("Benefit: Holdout 20% remains independent for evaluation")

        # Get training data (all rows where split_ext != 1)
        self.logger.debug("Extracting training data from pipeline.data.X_all")

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()

        if not hasattr(pipeline, 'data'):
            self.logger.error("No data attribute found in pipeline object")
            raise ValueError("Cannot find data. Ensure pipeline has been initialized with data.")

        if not hasattr(pipeline.data, 'X_all') or not hasattr(pipeline.data, 'y_all'):
            self.logger.error("No X_all or y_all found in pipeline.data")
            raise ValueError("Cannot find X_all/y_all. Ensure data has been loaded properly.")

        # Extract training data (exclude external validation set where split_ext == 1)
        self.logger.debug("Filtering for training data where split_ext != 1")
        X_train = pipeline.data.X_all[pipeline.data.X_all.split_ext != 1].copy()
        y_train = pipeline.data.y_all.loc[X_train.index].copy()

        # Verify split_int exists for CV folds
        if 'split_int' not in X_train.columns:
            self.logger.error("split_int column not found in X_train")
            raise ValueError(
                "split_int column not found. This is needed to define CV folds. "
                "Ensure your data has been properly prepared with CV fold assignments."
            )

        # Verify we have the expected number of folds
        unique_folds = X_train['split_int'].nunique()
        if unique_folds != n_folds:
            self.logger.warning(
                f"Data has {unique_folds} unique fold values in split_int, "
                f"but n_folds={n_folds} was specified. Using {unique_folds} folds."
            )
            n_folds = unique_folds

        self.logger.info(f"Training data extracted: {len(X_train)} samples across {n_folds} folds")

        # Transform and prepare data
        self.logger.debug("Transforming training data with OHE")
        X_train_transformed = pipeline.ohe.transform(X_train)
        y_train_array = self._extract_target_array(y_train)

        # Check prevalence and sample size
        n_events = int(y_train_array.sum())
        prevalence = y_train_array.mean()

        self.logger.info(
            f"Training set statistics: {len(y_train_array)} samples, "
            f"{n_events} events, prevalence={prevalence:.4f}"
        )

        print(f"\nTraining set: {len(y_train_array)} samples, {n_events} events ({prevalence:.3%})")
        print(f"Using split_int for {n_folds}-fold cross-validation")

        if n_events < 200:
            self.logger.warning(
                f"Training set has {n_events} events. "
                f"Van Calster et al. recommend ≥200 events for stable calibration."
            )
            print(f"⚠️  Note: Training set has {n_events} events.")
            print(f"   Van Calster et al. recommend ≥200 events for stable calibration.")
            print(f"   Results should be interpreted cautiously.")

        if prevalence < 0.05 and method == 'isotonic':
            self.logger.warning(
                f"Low prevalence ({prevalence:.4f}) with isotonic calibration may be unstable"
            )
            warnings.warn(
                f"⚠️  Event prevalence is {prevalence:.3%}. "
                f"Isotonic calibration may be unstable for rare events. "
                f"Consider method='sigmoid'.",
                UserWarning
            )

        # Get estimators and calibrate using CV based on split_int
        estimators = self._get_estimators()
        self.logger.info(f"Retrieved {len(estimators)} estimators for calibration")
        print(f"\nCalibrating {len(estimators)} models using {method} with {n_folds}-fold CV (based on split_int)...")

        # Create custom CV splitter based on split_int column
        from sklearn.model_selection import PredefinedSplit

        # Create fold indices for PredefinedSplit (-1 means exclude from any fold, used for test)
        # split_int values define which fold each sample belongs to
        fold_indices = X_train['split_int'].values - 1  # Assuming split_int is 1-indexed, convert to 0-indexed
        cv_splitter = PredefinedSplit(test_fold=fold_indices)

        self.logger.debug(f"Created PredefinedSplit CV splitter with {n_folds} folds based on split_int")

        self.calibrated_models = []
        for idx, model in enumerate(estimators):
            self.logger.debug(f"Calibrating model {idx+1}/{len(estimators)} with CV")
            base_estimator = self._get_base_estimator(model)

            # Use PredefinedSplit CV based on split_int column
            calibrated_clf = CalibratedClassifierCV(
                estimator=base_estimator,
                method=method,
                cv=cv_splitter,  # Use predefined splits from split_int
                ensemble=True  # Average predictions from all CV folds
            )

            # Fit on full training data with CV defined by split_int
            calibrated_clf.fit(X_train_transformed, y_train_array)
            self.calibrated_models.append(calibrated_clf)
            self.logger.debug(f"Model {idx+1} calibrated successfully with CV based on split_int")

        self.is_calibrated = True
        self.calibrated_on = '5fold_train_cv'
        self.calibration_method = f'{method}_cv{n_folds}'
        self.calibration_cv = n_folds

        self.logger.info(f"All {len(self.calibrated_models)} models calibrated with CV")
        print(f"✓ Successfully calibrated all {len(self.calibrated_models)} models")
        print(f"  Method: {method} with {n_folds}-fold cross-validation")

        # Store calibration info
        self.calibration_metrics['calibration_set_size'] = len(y_train_array)
        self.calibration_metrics['calibration_events'] = n_events
        self.calibration_metrics['calibration_prevalence'] = prevalence
        self.calibration_metrics['method'] = f'{method}_cv{n_folds}'
        self.calibration_metrics['cv_folds'] = n_folds
        self.calibration_metrics['calibration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.debug(f"Calibration metrics stored: {self.calibration_metrics}")

        print(f"\n{'='*80}")
        print("NEXT STEPS:")
        print(f"{'='*80}")
        print("1. Evaluate calibration on holdout set:")
        print("   calibrator.evaluate_on_holdout()")
        print("\n2. Plot calibration curves:")
        print("   calibrator.plot_calibration_curve()")
        print("\n3. Test on external dataset:")
        print("   calibrator.evaluate_on_external(X_external, y_external)")
        print(f"{'='*80}\n")


        # # Optionally save pipeline after calibration
        # if save:
        #     try:
        #         self.logger.info("Saving pipeline after cross-validation calibration")
        #         pipeline.save_Pipeline(compress=save_compress)
        #     except Exception as e:
        #         self.logger.exception("Failed to save pipeline after cross-validation calibration: %s", e)

        return self





    def calibrate_on_holdout(self, method='sigmoid', cv=5, X_holdout=None, y_holdout=None, save: bool = True, save_compress=('zlib', 3)):

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        print(pipeline.name)

        """Calibrate models using the holdout/validation set."""
        self.logger.info("="*80)
        self.logger.info("Starting model calibration on holdout/validation set")
        self.logger.info(f"Method: {method}, CV: {cv}")

        # Get holdout data
        if X_holdout is None or y_holdout is None:
            self.logger.debug("No holdout data provided, retrieving from pipeline")
            X_holdout, y_holdout = self._get_holdout_data()
        else:
            self.logger.debug("Using provided holdout data")

        # User-facing output
        print(f"\n{'='*80}")
        print("CALIBRATING MODELS ON HOLDOUT/VALIDATION SET")
        print(f"{'='*80}")
        print("Note: Using the independent 20% holdout set (X_val) for calibration")

        # Transform holdout data
        self.logger.debug("Transforming holdout data with OHE")
        X_holdout_transformed = pipeline.ohe.transform(X_holdout)
        y_holdout_array = self._extract_target_array(y_holdout)

        # Check prevalence and sample size
        n_events = int(y_holdout_array.sum())
        prevalence = y_holdout_array.mean()

        self.logger.info(
            f"Calibration set statistics: {len(y_holdout_array)} samples, "
            f"{n_events} events, prevalence={prevalence:.4f}"
        )

        print(f"Calibration set: {len(y_holdout_array)} samples, {n_events} events ({prevalence:.3%})")

        if n_events < 200:
            self.logger.warning(
                f"Low event count for calibration: {n_events} events. "
                f"Van Calster et al. recommend ≥200 events for stable curves."
            )
            print(f"⚠️  Note: Only {n_events} events available for calibration.")
            print(f"   Van Calster et al. recommend ≥200 events for stable calibration curves.")
            print(f"   Results should be interpreted cautiously.")

        if prevalence < 0.05 and method == 'isotonic':
            self.logger.warning(
                f"Low prevalence ({prevalence:.4f}) with isotonic calibration may be unstable"
            )
            warnings.warn(
                f"⚠️  Event prevalence is {prevalence:.3%}. "
                f"Isotonic calibration may be unstable for rare events. "
                f"Consider method='sigmoid'.",
                UserWarning
            )

        # Get estimators and calibrate
        estimators = self._get_estimators()
        self.logger.info(f"Retrieved {len(estimators)} estimators for calibration")
        print(f"\nCalibrating {len(estimators)} models using {method} method...")

        self.calibrated_models = []
        for idx, model in enumerate(estimators):
            self.logger.debug(f"Calibrating model {idx+1}/{len(estimators)}")
            base_estimator = self._get_base_estimator(model)
            calibrated_clf = CalibratedClassifierCV(
                estimator=base_estimator,
                method=method,
                cv='prefit',
                ensemble=True
            )
            calibrated_clf.fit(X_holdout_transformed, y_holdout_array)
            self.calibrated_models.append(calibrated_clf)
            self.logger.debug(f"Model {idx+1} calibrated successfully")

        self.is_calibrated = True
        self.calibrated_on = 'holdout_test'
        self.calibration_method = method
        self.calibration_cv = cv

        self.logger.info(f"All {len(self.calibrated_models)} models calibrated successfully")
        print(f"✓ Successfully calibrated all {len(self.calibrated_models)} models")

        # Store calibration info
        self.calibration_metrics['calibration_set_size'] = len(y_holdout_array)
        self.calibration_metrics['calibration_events'] = n_events
        self.calibration_metrics['calibration_prevalence'] = prevalence
        self.calibration_metrics['method'] = method
        self.calibration_metrics['calibration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.debug(f"Calibration metrics stored: {self.calibration_metrics}")




        return self




    def _get_holdout_data(self):

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        """Get holdout/validation data from pipeline object."""
        self.logger.debug("Attempting to retrieve holdout data from pipeline object")

        if hasattr(pipeline, 'data'):
            if hasattr(pipeline.data, 'X_val') and pipeline.data.X_val is not None:
                self.logger.debug("Found holdout data in pipeline.data.X_val")
                return pipeline.data.X_val, pipeline.data.y_val

        if hasattr(pipeline, 'pipeline') and hasattr(pipeline, 'data'):
            if hasattr(pipeline.data, 'X_val'):
                self.logger.debug("Found holdout data in pipeline.data.X_val")
                return pipeline.data.X_val, pipeline.data.y_val

        self.logger.error("No holdout/validation data (X_val) found in pipeline object")
        raise ValueError("No holdout/validation data (X_val) available.")

    def _extract_target_array(self, y_data):

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        """Extract target array from various y_data formats."""
        if isinstance(y_data, pd.Series):
            self.logger.debug("Extracted target from pandas Series")
            return y_data.values
        elif isinstance(y_data, pd.DataFrame):
            for col in ['status', 'status_cancerreg',
                       getattr(pipeline.user_input, 'target', None),
                       getattr(pipeline.user_input, 'target_to_validate_on', None)]:
                if col and col in y_data.columns:
                    self.logger.debug(f"Extracted target from DataFrame column: {col}")
                    return y_data[col].values
            self.logger.debug("Extracted target from first DataFrame column")
            return y_data.iloc[:, 0].values
        else:
            self.logger.debug("Converted target to numpy array")
            return np.asarray(y_data)

    def _get_estimators(self):

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        """Get list of estimators from pipeline object."""
        if hasattr(pipeline, 'list_estimators'):
            self.logger.debug("Retrieved estimators from pipeline.list_estimators")
            return pipeline.list_estimators
        elif hasattr(pipeline, 'master_RFC') and hasattr(pipeline.master_RFC, 'models'):
            self.logger.debug("Retrieved estimators from pipeline.master_RFC.models")
            return pipeline.master_RFC.models
        else:
            self.logger.error("Could not find estimators in pipeline object")
            raise ValueError("Could not find estimators in pipeline object")

    def _get_base_estimator(self, model):
        """Extract base estimator from model."""
        return model.best_estimator_ if hasattr(model, 'best_estimator_') else model

    def predict_proba_calibrated(self, X):
        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        """Get calibrated probability predictions."""
        if not self.is_calibrated:
            self.logger.error("Attempted prediction with uncalibrated models")
            raise ValueError("Models not yet calibrated. Run calibrate_on_holdout() first.")

        self.logger.debug(f"Generating calibrated predictions for {len(X)} samples")
        X_transformed = pipeline.ohe.transform(X)
        predictions = [model.predict_proba(X_transformed)[:, 1]
                      for model in self.calibrated_models]
        return np.mean(predictions, axis=0)

    def _calculate_calibration_slope_intercept(self, y_pred, y_true, use_class_weight=True):
        """
        Calculate calibration slope and intercept.

        Parameters:
        -----------
        y_pred : array-like
            Predicted probabilities
        y_true : array-like
            True binary labels
        use_class_weight : bool, default=True
            If True, use balanced class weights to handle imbalanced data.
            Recommended when positive class is overrepresented or when your
            base model (RFC) uses balanced subsampling.
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        logit_pred = np.log(y_pred_clipped / (1 - y_pred_clipped))

        # Use class_weight='balanced' to handle imbalanced data
        # This gives equal importance to both classes regardless of their frequency
        if use_class_weight:
            lr = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced'  # Key addition for imbalanced data
            )
        else:
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)

        lr.fit(logit_pred.reshape(-1, 1), y_true)

        slope, intercept = lr.coef_[0][0], lr.intercept_[0]

        if use_class_weight:
            self.logger.info(
                f"Calculated calibration slope={slope:.4f}, intercept={intercept:.4f} "
                f"(using balanced class weights)"
            )
        else:
            self.logger.info(f"Calculated calibration slope={slope:.4f}, intercept={intercept:.4f}")

        return slope, intercept


    #TODO: Unify the evaluation functions into one with a parameter for dataset



    def evaluate_on_external(self, X_external, y_external, dataset_name="external"):
        """Evaluate calibration on external validation data."""
        self.logger.info(f"Starting evaluation on external dataset: {dataset_name}")
        return self._evaluate_calibration_internal(
            X_external, y_external, dataset_name=dataset_name
        )

    def evaluate_on_holdout(self):
        """Evaluate calibration on the holdout/validation set."""
        self.logger.info("Starting evaluation on holdout set")
        X_holdout, y_holdout = self._get_holdout_data()
        return self._evaluate_calibration_internal(
            X_holdout, y_holdout, dataset_name="holdout (X_val)"
        )

    def _evaluate_calibration_internal(self, X_eval, y_eval, dataset_name):
        """Internal method to evaluate calibration."""
        if not self.is_calibrated:
            self.logger.error("Attempted evaluation with uncalibrated models")
            raise ValueError("Models not yet calibrated. Run calibrate_on_holdout() first.")

        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()

        self.logger.debug(f"Extracting target and transforming features for {dataset_name}")
        y_true = self._extract_target_array(y_eval)
        X_transformed = pipeline.ohe.transform(X_eval)

        self.logger.debug("Generating predictions for uncalibrated and calibrated models")
        y_proba_uncal = pipeline.master_RFC.predict_proba(X_transformed)
        y_proba_cal = self.predict_proba_calibrated(X_eval)

        # Calculate metrics - USE THE BALANCED FLAG
        self.logger.debug("Calculating evaluation metrics")
        auc_uncal = roc_auc_score(y_true, y_proba_uncal)
        auc_cal = roc_auc_score(y_true, y_proba_cal)

        # Pass the use_balanced_calibration flag to the slope/intercept calculation
        slope_uncal, intercept_uncal = self._calculate_calibration_slope_intercept(
            y_proba_uncal, y_true, use_class_weight=self.use_balanced_calibration
        )
        slope_cal, intercept_cal = self._calculate_calibration_slope_intercept(
            y_proba_cal, y_true, use_class_weight=self.use_balanced_calibration
        )

        brier_uncal = brier_score_loss(y_true, y_proba_uncal)
        brier_cal = brier_score_loss(y_true, y_proba_cal)
        logloss_uncal = log_loss(y_true, y_proba_uncal)
        logloss_cal = log_loss(y_true, y_proba_cal)

        prevalence = y_true.mean()
        n_events = int(y_true.sum())

        self.logger.info(
            f"Evaluation metrics for {dataset_name}: "
            f"AUC {auc_uncal:.4f}→{auc_cal:.4f}, "
            f"Brier {brier_uncal:.6f}→{brier_cal:.6f}, "
            f"Slope {slope_uncal:.4f}→{slope_cal:.4f}, "
            f"Intercept {intercept_uncal:+.4f}→{intercept_cal:+.4f}"
        )

        # Print results (user-facing output)
        self._print_evaluation_results(
            dataset_name, y_true, n_events, prevalence,
            auc_uncal, auc_cal, slope_uncal, slope_cal,
            intercept_uncal, intercept_cal, brier_uncal, brier_cal,
            logloss_uncal, logloss_cal
        )

        # Store results
        results = {
            'dataset': dataset_name,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(y_true),
            'n_events': n_events,
            'prevalence': prevalence,
            'auc_uncalibrated': auc_uncal,
            'auc_calibrated': auc_cal,
            'auc_change': auc_cal - auc_uncal,
            'brier_uncalibrated': brier_uncal,
            'brier_calibrated': brier_cal,
            'brier_improvement_pct': (brier_uncal - brier_cal) / brier_uncal * 100 if brier_uncal > 0 else 0,
            'logloss_uncalibrated': logloss_uncal,
            'logloss_calibrated': logloss_cal,
            'logloss_improvement_pct': (logloss_uncal - logloss_cal) / logloss_uncal * 100 if logloss_uncal > 0 else 0,
            'slope_uncalibrated': slope_uncal,
            'slope_calibrated': slope_cal,
            'intercept_uncalibrated': intercept_uncal,
            'intercept_calibrated': intercept_cal,
            'calibration_method': self.calibration_method,
        }

        self.calibration_metrics[f'{dataset_name}_evaluation'] = results
        self.evaluation_history.append(results)

        self.logger.debug(f"Stored evaluation results for {dataset_name}")

        return results

    def _print_evaluation_results(self, dataset_name, y_true, n_events, prevalence,
                                  auc_uncal, auc_cal, slope_uncal, slope_cal,
                                  intercept_uncal, intercept_cal, brier_uncal, brier_cal,
                                  logloss_uncal, logloss_cal):
        """
        Print comprehensive evaluation results.

        Note: This uses print() for user-facing formatted output.
        Internal operations use self.logger instead.
        """
        print("\n" + "="*80)
        print(f"CALIBRATION EVALUATION ON {dataset_name.upper()}")
        print("="*80)
        print(f"Dataset: {len(y_true)} samples, {n_events} events (prevalence: {prevalence:.4f})")

        print("\n" + "-"*80)
        print("1. DISCRIMINATION PERFORMANCE (MUST BE PRESERVED)")
        print("-"*80)
        print(f"Uncalibrated AUC: {auc_uncal:.4f}")
        print(f"Calibrated AUC:   {auc_cal:.4f}")
        print(f"Change:           {auc_cal - auc_uncal:+.4f}")

        auc_change = auc_cal - auc_uncal
        if abs(auc_change) > 0.02:
            self.logger.warning(
                f"AUC change exceeds threshold on {dataset_name}: "
                f"{auc_change:+.4f} (threshold: 0.02)"
            )
            print(f"\n⚠️  WARNING: AUC changed by {auc_change:+.4f}")
            print("   Calibration may have compromised discrimination!")
        else:
            print(f"\n✓ Discrimination preserved (acceptable change < 0.02)")

        print("\n" + "-"*80)
        print("2. WEAK CALIBRATION (Van Calster et al.)")
        print("-"*80)
        print(f"\nCalibration Intercept (target = 0):")
        print(f"  Uncalibrated: {intercept_uncal:+.4f}")
        print(f"  Calibrated:   {intercept_cal:+.4f}")
        if abs(intercept_cal) < 0.1:
            print("  → Good calibration-in-the-large ✓")

        print(f"\nCalibration Slope (target = 1):")
        print(f"  Uncalibrated: {slope_uncal:.4f}")
        print(f"  Calibrated:   {slope_cal:.4f}")
        if 0.9 <= slope_cal <= 1.1:
            print("  → Good spread ✓")

        print("\n" + "-"*80)
        print("3. OVERALL CALIBRATION METRICS")
        print("-"*80)
        print(f"\nBrier Score (lower is better):")
        print(f"  Uncalibrated: {brier_uncal:.6f}")
        print(f"  Calibrated:   {brier_cal:.6f}")
        brier_improvement = (brier_uncal - brier_cal) / brier_uncal * 100 if brier_uncal > 0 else 0
        print(f"  Improvement:  {brier_improvement:+.1f}%")

        print(f"\nLog Loss (lower is better):")
        print(f"  Uncalibrated: {logloss_uncal:.6f}")
        print(f"  Calibrated:   {logloss_cal:.6f}")
        logloss_improvement = (logloss_uncal - logloss_cal) / logloss_uncal * 100 if logloss_uncal > 0 else 0
        print(f"  Improvement:  {logloss_improvement:+.1f}%")

        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        if abs(auc_change) > 0.02:
            print(f"⚠️  CONCERNING: Discrimination was compromised on {dataset_name}")
        elif abs(intercept_cal) < 0.1 and 0.9 <= slope_cal <= 1.1:
            print(f"✓ EXCELLENT: Calibration generalizes well to {dataset_name}")
        elif abs(intercept_cal) < 0.2 and 0.8 <= slope_cal <= 1.2:
            print(f"✓ GOOD: Calibration shows reasonable generalization to {dataset_name}")
        else:
            print(f"⚠  MODERATE: Calibration shows limited generalization to {dataset_name}")
        print("="*80)

    def plot_calibration_curve(self, X_eval=None, y_eval=None, dataset_name="external", fontsize=12,
                            save=True, use_clinical_viz=True,
                            colors: Optional[Dict[str, str]] = None, title=True, add_letters=True):
        """
        Plot calibration curves with option for clinical visualization.

        Parameters:
        -----------
        use_clinical_viz : bool, default=True
            If True and event is rare, use comprehensive clinical visualization

        fontsize : int, default=12
            Font size for all text in the plots.
        """
        if not self.is_calibrated:
            self.logger.error("Attempted plotting with uncalibrated models")
            raise ValueError("Models not yet calibrated. Run calibrate_on_holdout() first.")

        self.logger.info(f"Plotting calibration curve for dataset: {dataset_name}")
        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()

        if X_eval is None or y_eval is None:
            self.logger.debug("No evaluation data provided, using holdout data")
            X_eval, y_eval = self._get_holdout_data()
            if dataset_name == "external":
                dataset_name = "holdout (X_val)"

        y_true = self._extract_target_array(y_eval)
        X_transformed = pipeline.ohe.transform(X_eval)
        y_proba_uncal = pipeline.master_RFC.predict_proba(X_transformed)
        y_proba_cal = self.predict_proba_calibrated(X_eval)

        # merge defaults with per-call overrides
        colors = {**self.default_plot_colors, **(colors or {})}
        color_prior = colors['color_prior']
        edgecolor_prior = colors['edgecolor_prior']
        color_past = colors['color_past']
        edgecolor_past = colors['edgecolor_past']


        prevalence = y_true.mean()
        is_rare_event = prevalence < 0.05

        self.logger.info(
            f"Dataset prevalence: {prevalence:.4f}, "
            f"rare_event={is_rare_event}, use_clinical_viz={use_clinical_viz}"
        )

        # Use clinical visualization for rare events
        if is_rare_event and use_clinical_viz:
            self.logger.debug("Using clinical visualization for rare event")
            fig = self._plot_clinical_calibration_assessment(
                y_true, y_proba_uncal, y_proba_cal, dataset_name,
                fontsize=fontsize, colors=colors, title=title, add_letters=add_letters, save=save
            )
            return fig

        # Standard visualization for common events
        self.logger.debug("Using standard calibration curve visualization")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        self._plot_calibration_curve_ax(
            ax1, y_true, y_proba_uncal,
            f'Before Calibration\n{dataset_name.capitalize()}', color_prior, fontsize=fontsize
        )
        self._plot_calibration_curve_ax(
            ax2, y_true, y_proba_cal,
            f'After Calibration ({self.calibration_method})\n{dataset_name.capitalize()}', color_past, fontsize=fontsize
        )

        fig.suptitle(f'Calibration Curves (Prevalence: {prevalence:.2%})',
                    fontsize=fontsize, y=0.98)
        plt.tight_layout()

        if save:
            save_path = os.path.join(
                getattr(pipeline.user_input, 'fig_path', '.'),
                f"Calibration_{pipeline.user_input.col_subset}_{pipeline.user_input.row_subset}_{dataset_name}_{self.calibration_method}"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}.svg', format='svg', bbox_inches='tight')
        plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=600)
        self.logger.info(f"Calibration curve saved to: {save_path}")
        print(f"\nCalibration curve saved to: {save_path}")

        return fig



    def _plot_clinical_calibration_assessment(self, y_true, y_pred_uncal, y_pred_cal,
                                            dataset_name, fontsize=12,
                                            colors: Optional[Dict[str,str]] = None, title=True, add_letters=True, save=True):
        """Create comprehensive clinical visualization for rare events."""
        self.logger.debug("Creating clinical calibration assessment visualization")
        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()

        colors = {**self.default_plot_colors, **(colors or {})}
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.35, top=0.87, bottom=0.05)

        # Add a bold column title row above the two main columns
        # x=0.25 and x=0.75 center the labels over left/right columns; adjust y to tune spacing
        fig.text(0.285, 0.915, "Metrics prior to calibration", ha="center", va="bottom",
                 fontsize=fontsize*1.1, fontweight="bold")
        fig.text(0.735, 0.915, "Metrics after calibration", ha="center", va="bottom",
                 fontsize=fontsize*1.1, fontweight="bold")

        prevalence = y_true.mean()
        n_events = int(y_true.sum())
        n_controls = len(y_true) - n_events

        # 1a. PREDICTION DISTRIBUTION COMPARISON (Top Row, left)
        self.logger.debug("Plotting prediction distributions")
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot histograms for uncalibrated predictions
        n_controls_hist, bins_controls, _ = ax1.hist(y_pred_uncal[y_true == 0], bins=50, alpha=0.5,
        color=colors.get('control_fill'), label=f'Controls (n={n_controls})',
        density=True, edgecolor=colors.get('control_edge'), linewidth=0.5)

        n_events_hist, bins_events, _ = ax1.hist(y_pred_uncal[y_true == 1], bins=50, alpha=0.9,
        color=colors.get('color_prior'), label=f'Cases (n={n_events})',
        density=True, edgecolor=colors.get('edgecolor_prior'), linewidth=0.9)

        # Add dotted line outlines for better visibility
        # Controls outline (grey dotted line)
        bin_centers_controls = (bins_controls[:-1] + bins_controls[1:]) / 2
        ax1.plot(bin_centers_controls, n_controls_hist,
                color=colors.get('control_edge'), linestyle='-', linewidth=1, alpha=0.8)

        # Cases outline (red dotted line), likely a bit messy due to fewer events
        #bin_centers_events = (bins_events[:-1] + bins_events[1:]) / 2
        #ax1.plot(bin_centers_events, n_events_hist,
        #        color=colors.get('color_prior'), linestyle='-', linewidth=1, alpha=0.9)


        ax1.axvline(prevalence, color='darkgrey', linestyle='--', linewidth=2.5,
                    label=f'True prevalence: {prevalence:.3%}', alpha=1)
        ax1.set_xlabel('Predicted Risk', fontsize=fontsize)
        ax1.set_ylabel('Density', fontsize=fontsize)
        ax1.set_title('Predicted risk distribution', fontsize=fontsize, pad=5)
        ax1.legend(fontsize=fontsize, frameon=False)
        ax1.tick_params(axis='both', which='both', direction='out', length=4, width=0.5,
                labelsize=fontsize, pad=5, top=False, right=False, color='black')
        for spine in ax1.spines.values():
            spine.set_linewidth(1)


        ax2 = fig.add_subplot(gs[0, 1])
        # Plot histograms for calibrated predictions
        n_controls_hist_cal, bins_controls_cal, _ = ax2.hist(y_pred_cal[y_true == 0], bins=50, alpha=0.5,
                color=colors.get('control_fill'), label=f'Controls (n={n_controls})',
                density=True, edgecolor='darkgrey', linewidth=0.5)
        n_events_hist_cal, bins_events_cal, _ = ax2.hist(y_pred_cal[y_true == 1], bins=50, alpha=0.9,
                color=colors.get('color_past'), label=f'Cases (n={n_events})',
                density=True, edgecolor=colors.get('edgecolor_past'), linewidth=0.5)
        # Add dotted line outlines for better visibility
        # Controls outline (grey dotted line)
        bin_centers_controls_cal = (bins_controls_cal[:-1] + bins_controls_cal[1:]) / 2
        ax2.plot(bin_centers_controls_cal, n_controls_hist_cal,
                color='darkgrey', linestyle='-', linewidth=1, alpha=0.8)
        # Cases outline (blue dotted line), likely a bit messy due to fewer events
        #bin_centers_events_cal = (bins_events_cal[:-1] + bins_events_cal[1:]) / 2
        #ax2.plot(bin_centers_events_cal, n_events_hist_cal,
        #        color=colors.get('color_past'), linestyle='-', linewidth=1, alpha=0.9)
        ax2.axvline(prevalence, color='darkgrey', linestyle='--', linewidth=2.5,
                    label=f'True prevalence: {prevalence:.3%}', alpha=1)
        ax2.set_xlabel('Predicted Risk', fontsize=fontsize)
        ax2.set_ylabel('Density', fontsize=fontsize)
        ax2.set_title('Predicted risk distribution', fontsize=fontsize, pad=5)
        ax2.legend(fontsize=fontsize, frameon=False)
        ax2.tick_params(axis='both', which='major', direction='out', length=4, width=0.5,
                labelsize=fontsize, pad=5, top=False, right=False, color='black')
        for spine in ax2.spines.values():
            spine.set_linewidth(1)

        # 2. DECILE-BASED CALIBRATION (Middle Row)
        self.logger.debug("Calculating and plotting decile-based calibration")
        def calculate_decile_calibration(y_true_arr, y_pred_arr):
            deciles = np.percentile(y_pred_arr, np.arange(10, 100, 10))
            decile_groups = np.digitize(y_pred_arr, deciles)
            results = []
            for i in range(10):
                mask = decile_groups == i
                if mask.sum() > 0:
                    n = mask.sum()
                    n_ev = y_true_arr[mask].sum()
                    mean_pred = y_pred_arr[mask].mean()
                    obs_rate = y_true_arr[mask].mean()
                    se = np.sqrt(obs_rate * (1 - obs_rate) / n) if n > 0 else 0
                    results.append({
                        'decile': i + 1, 'n': n, 'n_events': int(n_ev),
                        'mean_predicted': mean_pred, 'observed_rate': obs_rate,
                        'lower_ci': max(0, obs_rate - 1.96 * se),
                        'upper_ci': min(1, obs_rate + 1.96 * se)
                    })
            return pd.DataFrame(results)

        ax4 = fig.add_subplot(gs[1, 0])
        decile_uncal = calculate_decile_calibration(y_true, y_pred_uncal)
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=2.5, alpha=0.7,
                label='Perfect calibration', zorder=1, color='darkgrey')
        ax4.errorbar(decile_uncal['mean_predicted'], decile_uncal['observed_rate'],
                    yerr=[decile_uncal['observed_rate'] - decile_uncal['lower_ci'],
                        decile_uncal['upper_ci'] - decile_uncal['observed_rate']],
                    fmt='o-', color=colors.get('color_prior'), linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, label='Observed', alpha=0.85, zorder=2,
                    markeredgecolor=colors.get('edgecolor_prior'), markeredgewidth=0.5)
        # for idx, row in decile_uncal.iterrows():
        #     if idx % 2 == 0:
        #         ax4.annotate(f"{row['n_events']}/{row['n']}",
        #                     (row['mean_predicted'], row['observed_rate']),
        #                     textcoords="offset points", xytext=(0, 12),
        #                     ha='center', fontsize=fontsize*0.8, alpha=0.75)
        ax4.set_xlabel('Mean Predicted Risk', fontsize=fontsize, labelpad=5)
        ax4.set_ylabel('Observed Event Rate', fontsize=fontsize, labelpad=5)
        ax4.set_title('Calibration', fontsize=fontsize, pad=5)
        ax4.legend(fontsize=fontsize, frameon=False, loc='upper right')
        ax4.tick_params(axis='both', which='major', direction='out', length=4, width=0.5,
                labelsize=fontsize, pad=5, top=False, right=False, color='black')
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, max(0.05, decile_uncal['observed_rate'].max() * 1.2)])
        for spine in ax4.spines.values():
            spine.set_linewidth(1)

        ax5 = fig.add_subplot(gs[1, 1])
        decile_cal = calculate_decile_calibration(y_true, y_pred_cal)
        max_pred_cal = y_pred_cal.max()
        xlim_max = min(0.2, max_pred_cal * 1.3)
        ylim_max = max(prevalence * 3, decile_cal['observed_rate'].max() * 1.3)
        ax5.plot([0, xlim_max], [0, ylim_max], 'k--', linewidth=2.5, alpha=0.7,
                label='Perfect calibration', zorder=1, color='darkgrey')
        # ax5.axhline(prevalence, color='darkgrey', linestyle=':', linewidth=2,
        #             alpha=0.6, label=f'Prevalence: {prevalence:.2%}', zorder=1)
        ax5.errorbar(decile_cal['mean_predicted'], decile_cal['observed_rate'],
                    yerr=[decile_cal['observed_rate'] - decile_cal['lower_ci'],
                        decile_cal['upper_ci'] - decile_cal['observed_rate']],
                    fmt='o-', color=colors.get('color_past'), linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, label='Observed', alpha=0.85, zorder=2,
                    markeredgecolor=colors.get('edgecolor_past'), markeredgewidth=0.5)
        # for idx, row in decile_cal.iterrows():
        #     if row['mean_predicted'] <= xlim_max and idx % 2 == 0:
        #         ax5.annotate(f"{row['n_events']}/{row['n']}",
        #                     (row['mean_predicted'], row['observed_rate']),
        #                     textcoords="offset points", xytext=(0, 12),
        #                     ha='center', fontsize=fontsize*0.7, alpha=0.75)
        ax5.set_xlabel('Mean Predicted Risk', fontsize=fontsize, labelpad=5)
        ax5.set_ylabel('Observed Event Rate', fontsize=fontsize, labelpad=5)
        ax5.set_title('Calibration (Zoomed)', fontsize=fontsize, pad=5)
        ax5.legend(fontsize=fontsize, frameon=False, loc='lower right')
        ax5.tick_params(axis='both', which='major', direction='out', length=4, width=0.5,
                labelsize=fontsize, pad=5, top=False, right=False, color='black')
        ax5.set_xlim([0, xlim_max])
        ax5.set_ylim([0, ylim_max])
        for spine in ax5.spines.values():
            spine.set_linewidth(1)

        # 3. RISK STRATIFICATION (Bottom Row) - spans both columns
        self.logger.debug("Plotting risk stratification analysis")
        ax7 = fig.add_subplot(gs[2, :])
        thresholds = np.percentile(y_pred_cal, [50, 75, 90, 95, 98])
        results_by_threshold = []
        for pct, threshold in zip([50, 75, 90, 95, 98], thresholds):
            mask_uncal = y_pred_uncal >= np.percentile(y_pred_uncal, pct)
            mask_cal = y_pred_cal >= threshold
            n_flagged_uncal = mask_uncal.sum()
            n_flagged_cal = mask_cal.sum()
            events_uncal = y_true[mask_uncal].sum()
            events_cal = y_true[mask_cal].sum()
            results_by_threshold.append({
                'Threshold': f'≥{pct}th %ile',
                'Before_N': n_flagged_uncal,
                'Before_Events': int(events_uncal),
                'Before_PPV': events_uncal / n_flagged_uncal if n_flagged_uncal > 0 else 0,
                'After_N': n_flagged_cal,
                'After_Events': int(events_cal),
                'After_PPV': events_cal / n_flagged_cal if n_flagged_cal > 0 else 0,
            })
        results_df = pd.DataFrame(results_by_threshold)
        x = np.arange(len(results_df))
        width = 0.35
        ax7.bar(x - width/2, results_df['Before_PPV'] * 100, width,
                label='Before Calibration', color=colors.get('color_prior', colors.get('before_bar')), alpha=0.8,
                edgecolor=colors.get('edgecolor_prior'), linewidth=0.5)
        ax7.bar(x + width/2, results_df['After_PPV'] * 100, width,
                label='After Calibration', color=colors.get('color_past', colors.get('after_bar')), alpha=0.8,
                edgecolor=colors.get('edgecolor_past'), linewidth=0.5)
        ax7.axhline(prevalence * 100, color='darkgrey', linestyle='--', linewidth=2,
                    label=f'Baseline prevalence: {prevalence:.2%}', alpha=0.7)
        ax7.set_xlabel('Risk Threshold', fontsize=fontsize)
        ax7.set_ylabel('Positive Predictive Value (%)', fontsize=fontsize)
        ax7.set_title('Threshold-dependent PPV',
                    fontsize=fontsize)
        ax7.set_xticks(x)
        ax7.set_xticklabels(results_df['Threshold'])
        ax7.legend(fontsize=fontsize, frameon=False)
        ax7.tick_params(axis='both', which='major', direction='out', length=4, width=0.5,
                labelsize=fontsize, pad=5, top=False, right=False, color='black')
        for spine in ax7.spines.values():
            spine.set_linewidth(1)

        # Remove top spine to prevent label cutoff
        ax7.spines['top'].set_visible(False)

        for i, row in results_df.iterrows():
            ax7.text(i - width/2, row['Before_PPV'] * 100 + 0.05,
                    f"{row['Before_Events']}/\n{row['Before_N']}",
                    ha='center', va='bottom', fontsize=fontsize)
            ax7.text(i + width/2, row['After_PPV'] * 100 + 0.05,
                    f"{row['After_Events']}/\n{row['After_N']}",
                    ha='center', va='bottom', fontsize=fontsize)

        if title:
            fig.suptitle(f'Calibration of prediction models on disease prevalence\n' +
                        f'Dataset: {dataset_name} (Prevalence: {prevalence:.2%}, N={len(y_true):,}, Events={n_events})',
                        fontsize=fontsize*1.2, y=0.995)
        # Add panel letters (compact section)
        if add_letters:
            letter_config = [
                (ax1, 'a'),  # Top left
                (ax2, 'b'),  # Top right
                (ax4, 'c'),  # Middle left
                (ax5, 'd'),  # Middle right

            ]

            for ax, letter in letter_config:
                ax.text(-0.13, 1.15, letter, transform=ax.transAxes,
                    fontsize=fontsize*2, fontweight='bold',
                    va='top', ha='right')

            #Handle e separately for full-width bottom plot
            ax7.text(-0.05, 1.2, "e", transform=ax7.transAxes,
                    fontsize=fontsize*2, fontweight='bold',
                    va='top', ha='right')


        if save:
            save_path = os.path.join(
                getattr(pipeline.user_input, 'fig_path', '.'),
                f"Calibration_{pipeline.user_input.col_subset}_{pipeline.user_input.row_subset}_{dataset_name}_{self.calibration_method}"
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f'{save_path}.svg', format='svg', bbox_inches='tight')
            plt.savefig(f'{save_path}.png', format='png', bbox_inches='tight', dpi=600)
            self.logger.info(f"Clinical calibration figure saved as SVG and PNG to: {save_path}")


        plt.tight_layout()
        plt.show()
        return fig


    def _plot_calibration_curve_ax(self, ax, y_true, y_pred, title, color, fontsize=12):
        """Plot smooth calibration curve on given axis."""
        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        window = max(50, len(y_pred) // 30)
        window = min(window, len(y_pred) // 3)
        if window % 2 == 0:
            window += 1
        window = max(3, window)

        y_smoothed = np.convolve(y_true_sorted, np.ones(window)/window, mode='valid')
        x_smoothed = y_pred_sorted[window//2:-(window//2)]

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration',
                linewidth=2, alpha=0.7)
        ax.plot(x_smoothed, y_smoothed, color=color, linewidth=2.5,
                label='Model', alpha=0.85)

        prevalence = y_true.mean()
        ax.axhline(y=prevalence, color='darkgrey', linestyle=':',
                linewidth=1.5, alpha=0.6,
                label=f'Prevalence: {prevalence:.4f}')

        ax.hist(y_pred, bins=min(50, len(np.unique(y_pred))),
                alpha=0.3, color=color, edgecolor='none', range=(0, 1))

        ax.set_xlabel('Predicted Probability', fontsize=fontsize)
        ax.set_ylabel('Observed Frequency', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize*1.1, pad=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=fontsize*0.8, loc='upper left', frameon=False)
        ax.tick_params(labelsize=fontsize*0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    def export_metrics_to_excel(self, filepath=None, model_name=None):
        """
        Export calibration metrics to Excel in a clean single-sheet format.

        Each row represents a model/dataset combination with all metrics in columns.
        Perfect for tracking and comparing multiple models over time.
        """
        if not self.is_calibrated:
            self.logger.error("Attempted to export metrics without calibration")
            raise ValueError("No calibration metrics to export. Run calibrate_on_holdout() first.")

        self.logger.info("Starting Excel export of calibration metrics")
        # get the pipeline object for internal reference
        pipeline = self._get_pipeline()

        # Get model name
        if model_name is None:
            if hasattr(pipeline, 'user_input'):
                model_name = getattr(pipeline.user_input, 'col_subset', 'Unknown_Model')
            elif hasattr(pipeline, 'name'):
                model_name = pipeline.name
            else:
                model_name = 'Unknown_Model'

        self.logger.debug(f"Using model name: {model_name}")

        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if hasattr(pipeline, 'user_input'):
                base_path = getattr(pipeline.user_input, 'model_path', '.')
                filepath = os.path.join(
                    base_path, 'calibration_reports',
                    f"calibration_tracking_{timestamp}.xlsx"
                )
            else:
                filepath = f"calibration_tracking_{timestamp}.xlsx"

        self.logger.info(f"Export filepath: {filepath}")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Build the master dataframe - one row per model/dataset combination
        rows = []

        if not self.evaluation_history:
            self.logger.warning("No evaluation history available for export")
            print("⚠️  No evaluations have been performed yet.")
            print("   Run evaluate_on_holdout() or evaluate_on_external() first.")
            return None

        self.logger.debug(f"Building dataframe from {len(self.evaluation_history)} evaluations")

        for eval_result in self.evaluation_history:
            # Create a comprehensive row with all metrics
            row = {
                'Model_Name': model_name,
                'Dataset': eval_result['dataset'],
                'Evaluation_Date': eval_result['evaluation_date'],
                'Calibration_Method': eval_result['calibration_method'],
                'Calibration_Date': self.calibration_metrics.get('calibration_date', 'N/A'),
                'N_Samples': eval_result['n_samples'],
                'N_Events': eval_result['n_events'],
                'Prevalence': eval_result['prevalence'],
                'AUC_Uncalibrated': eval_result['auc_uncalibrated'],
                'AUC_Calibrated': eval_result['auc_calibrated'],
                'AUC_Change': eval_result['auc_change'],
                'AUC_Preserved': 'Yes' if abs(eval_result['auc_change']) < 0.02 else 'No',
                'Brier_Uncalibrated': eval_result['brier_uncalibrated'],
                'Brier_Calibrated': eval_result['brier_calibrated'],
                'Brier_Improvement_%': eval_result['brier_improvement_pct'],
                'LogLoss_Uncalibrated': eval_result['logloss_uncalibrated'],
                'LogLoss_Calibrated': eval_result['logloss_calibrated'],
                'LogLoss_Improvement_%': eval_result['logloss_improvement_pct'],
                'Slope_Uncalibrated': eval_result['slope_uncalibrated'],
                'Slope_Calibrated': eval_result['slope_calibrated'],
                'Slope_OK': 'Yes' if 0.9 <= eval_result['slope_calibrated'] <= 1.1 else 'No',
                'Intercept_Uncalibrated': eval_result['intercept_uncalibrated'],
                'Intercept_Calibrated': eval_result['intercept_calibrated'],
                'Intercept_OK': 'Yes' if abs(eval_result['intercept_calibrated']) < 0.1 else 'No',
                'Overall_Quality': self._assess_calibration_quality(eval_result),
                'Recommendation': self._generate_single_recommendation(eval_result),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        column_order = [
            'Model_Name', 'Dataset', 'Evaluation_Date', 'Calibration_Method',
            'Calibration_Date', 'N_Samples', 'N_Events', 'Prevalence',
            'AUC_Uncalibrated', 'AUC_Calibrated', 'AUC_Change', 'AUC_Preserved',
            'Brier_Uncalibrated', 'Brier_Calibrated', 'Brier_Improvement_%',
            'LogLoss_Uncalibrated', 'LogLoss_Calibrated', 'LogLoss_Improvement_%',
            'Slope_Uncalibrated', 'Slope_Calibrated', 'Slope_OK',
            'Intercept_Uncalibrated', 'Intercept_Calibrated', 'Intercept_OK',
            'Overall_Quality', 'Recommendation'
        ]

        df = df[column_order]

        # Check if file exists and append if it does
        if os.path.exists(filepath):
            self.logger.info("Existing file found, attempting to merge data")
            print(f"\n📝 File exists. Checking for updates...")
            try:
                existing_df = pd.read_excel(filepath)
                self.logger.debug(f"Loaded existing file with {len(existing_df)} rows")

                existing_df = existing_df[~existing_df['Model_Name'].isin([model_name])]
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.sort_values(['Model_Name', 'Dataset', 'Evaluation_Date'])

                self.logger.info(f"Merged data: {len(df)} total rows")
                print(f"✓ Updated existing file with new {model_name} results")
            except Exception as e:
                self.logger.warning(f"Could not read existing file: {e}. Creating new file.")
                print(f"⚠️  Could not read existing file: {e}")
                print(f"   Creating new file instead...")

        # Save to Excel
        self.logger.debug("Writing DataFrame to Excel")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Calibration_Tracking', index=False)

            worksheet = writer.sheets['Calibration_Tracking']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

        self.logger.info(f"Successfully exported {len(df)} rows to {filepath}")

        print(f"\n{'='*80}")
        print("CALIBRATION METRICS EXPORTED TO EXCEL")
        print(f"{'='*80}")
        print(f"File: {filepath}")
        print(f"Model: {model_name}")
        print(f"Datasets evaluated: {len(self.evaluation_history)}")
        print(f"Total rows in file: {len(df)}")
        print(f"\n✓ All metrics for {model_name} are now in a single sheet")
        print(f"  Each row = Model + Dataset combination")
        print(f"  Each column = Metric (AUC, Brier, Slope, etc.)")
        print(f"{'='*80}\n")

        return filepath

    def _generate_single_recommendation(self, eval_result):
        """Generate a concise recommendation for a single evaluation."""
        auc_change = eval_result['auc_change']
        slope = eval_result['slope_calibrated']
        intercept = eval_result['intercept_calibrated']

        if abs(auc_change) > 0.02:
            return "CAUTION: Discrimination compromised - Review calibration approach"
        elif abs(intercept) < 0.1 and 0.9 <= slope <= 1.1:
            return "EXCELLENT: Use calibrated model clinically"
        elif abs(intercept) < 0.2 and 0.8 <= slope <= 1.2:
            return "GOOD: Suitable for clinical use - Monitor performance"
        else:
            return "MODERATE: Acceptable but monitor closely"

    def _assess_calibration_quality(self, eval_result):
        """Assess overall calibration quality for a single evaluation."""
        auc_preserved = abs(eval_result['auc_change']) < 0.02
        slope_good = 0.9 <= eval_result['slope_calibrated'] <= 1.1
        intercept_good = abs(eval_result['intercept_calibrated']) < 0.1

        if not auc_preserved:
            return 'Poor - Discrimination Compromised'
        elif slope_good and intercept_good:
            return 'Excellent'
        elif (0.8 <= eval_result['slope_calibrated'] <= 1.2 and
              abs(eval_result['intercept_calibrated']) < 0.2):
            return 'Good'
        else:
            return 'Moderate'

    def _generate_recommendations(self):
        """Generate recommendations for each evaluated dataset."""
        recommendations = {}
        for eval_result in self.evaluation_history:
            dataset = eval_result['dataset']
            recommendations[dataset] = self._generate_single_recommendation(eval_result)
        return recommendations

    def get_summary(self):
        """Get summary of calibration status and results."""
        summary = {
            'is_calibrated': self.is_calibrated,
            'calibration_method': self.calibration_method,
            'n_calibrated_models': len(self.calibrated_models) if self.is_calibrated else 0,
        }

        if self.is_calibrated:
            summary.update({
                'calibration_date': self.calibration_metrics.get('calibration_date'),
                'calibration_set_size': self.calibration_metrics.get('calibration_set_size'),
                'calibration_events': self.calibration_metrics.get('calibration_events'),
                'calibration_prevalence': self.calibration_metrics.get('calibration_prevalence'),
                'n_evaluations': len(self.evaluation_history),
            })

            for key, value in self.calibration_metrics.items():
                if key.endswith('_evaluation'):
                    summary[key] = value

        return summary

    def print_summary(self):
        """Print a formatted summary of calibration status."""
        self.logger.debug("Printing calibration summary")

        print("\n" + "="*80)
        print("CALIBRATION LAYER SUMMARY")
        print("="*80)

        if not self.is_calibrated:
            print("Status: Not calibrated")
            print("\nUse calibrate_on_holdout() to calibrate models.")
        else:
            print(f"Status: Calibrated")
            print(f"Method: {self.calibration_method}")
            print(f"Date: {self.calibration_metrics.get('calibration_date', 'N/A')}")
            print(f"Models calibrated: {len(self.calibrated_models)}")

            print(f"\nCalibration Set:")
            print(f"  Size: {self.calibration_metrics.get('calibration_set_size', 'N/A')}")
            print(f"  Events: {self.calibration_metrics.get('calibration_events', 'N/A')}")
            print(f"  Prevalence: {self.calibration_metrics.get('calibration_prevalence', 0):.4f}")

            if self.evaluation_history:
                print(f"\nEvaluations performed: {len(self.evaluation_history)}")
                print("\nDatasets evaluated:")
                datasets = set([e['dataset'] for e in self.evaluation_history])
                for dataset in datasets:
                    print(f"  - {dataset}")

        print("="*80 + "\n")