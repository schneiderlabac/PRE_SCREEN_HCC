import os
from joblib import dump
from modeling_pipeline.pipeline import Pipeline as _Pipeline
from modeling_pipeline.calibration import CalibrationLayer






class ExportExtVal(_Pipeline):
    """
    External Validation class for pipeline objects.
    Init should be called with a trained pipeline object in your training IDE
    Ext_val object can then be saved and loaded in a different IDE/environment for external validation.
    """
    def __init__(self, pl) -> None:
        self.user_input = pl.user_input
        self.master_RFC = pl.master_RFC
        self.list_estimators = [i.best_estimator_ for i in pl.master_RFC.models]
        self.name = pl.name
        self.ohe = pl.ohe
        self.mapper = pl.mapper
        self.columngroups_df = pl.data.columngroups_df
        self.calibration = pl.calibration if hasattr(pl, 'calibration') else None
        self.plots = pl.plots if hasattr(pl, 'plots') else None

        #self.calibration.parent = {} #Remove the parent subfolder which would contain pipeline data


        self.master_RFC.models_with_eids_of_datasets = {}
        # Make plots optional - only include if it exists
        self.plots = getattr(pl, 'plots', {})

        self.pipeline_output_path = "."

        # Clean the trained_model to remove problematic references
        self.trained_model = self._create_clean_trained_model(pl.trained_model)

    def _setup_module_mappings(self):
        """Create modeling_pipeline module structure to handle dependencies"""
        import sys
        import types

        if 'modeling_pipeline' not in sys.modules:
            # Create main module
            modeling_pipeline = types.ModuleType('modeling_pipeline')
            sys.modules['modeling_pipeline'] = modeling_pipeline

            # Create common submodules as dummies
            submodules = [
                'ablation', 'wrapper_roc_analysis', 'wrapper_violins_prcs',
                'export_tables', 'pp', 'plot', 'training', 'training.models',
                'training.train_test', 'pipeline'
            ]

            for submodule_name in submodules:
                dummy_module = types.ModuleType(f'modeling_pipeline.{submodule_name}')

                if '.' in submodule_name:
                    # Handle nested modules like training.models
                    parts = submodule_name.split('.')
                    current = modeling_pipeline
                    for i, part in enumerate(parts[:-1]):
                        if not hasattr(current, part):
                            setattr(current, part, types.ModuleType(f'modeling_pipeline.{".".join(parts[:i+1])}'))
                        current = getattr(current, part)
                    setattr(current, parts[-1], dummy_module)
                else:
                    setattr(modeling_pipeline, submodule_name, dummy_module)

                sys.modules[f'modeling_pipeline.{submodule_name}'] = dummy_module

    def _create_clean_trained_model(self, trained_model):
        """Create a simple dictionary-based version of trained_model"""
        clean_data = {}

        # Extract model_with_info as a simple dictionary
        if hasattr(trained_model, 'model_with_info'):
            clean_model_with_info = {}
            for key, value_orig in trained_model.model_with_info.items():
                clean_model_with_info[key] = {"model": value_orig.get("model")}
            clean_data['model_with_info'] = clean_model_with_info

        # Extract models if they exist
        if hasattr(trained_model, 'models'):
            clean_data['models'] = trained_model.models

        return clean_data

    def save(self, path: str = ".", compress=('zlib', 3)):


        # Ensure module mappings exist before saving
        self._setup_module_mappings()

        save_dir = os.path.join(self.user_input.path, "Models", "Validation_Objects")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{self.name}_external_val.joblib")
        dump(self, file_path,compress=compress)

        try:
            dump(self, file_path, compress=compress)
            print(f"External validation object has been saved to: {os.path.abspath(file_path)}")
        except Exception as e:
            print(f"Error saving: {e}")
            # Try without compression as fallback
            try:
                dump(self, file_path, compress=None)
                print(f"External validation object saved (uncompressed) to: {os.path.abspath(file_path)}")
            except Exception as e2:
                print(f"Failed to save: {e2}")
                raise