# Overview

hcc includes the relevant files for training and UKB testing.

hcc_ext_val includes the relevant files for external inference/testing. Start with file 01, load a dataset, load a joblib model, and run the relevant notebook chunks, especially:

# Set paths
model_name = f'Validation_Objects/Pipeline_{DOI}_{row_subset}_Model_{model}_{estimator}_external_val.joblib'
full_path = os.path.join(path, model_name)
ext_val = load(full_path) # Load the file
print(f"Loading file from: {full_path}")


# Initialize pipeline
pl_ext={} #reset pipeline before creating new one
pl_ext=Pipeline(ext_val_obj=ext_val) #Initialize pipeline object
pl_ext.external_validation(X_val=X_val_df,y_val=y_val_df)  # Load ext_data into pipeline
class fix_trained_model:
    def __init__(self, trained_model_self):
        for k, v in trained_model_self.items():
            setattr(self, k, v)
pl_ext.trained_model = fix_trained_model(pl_ext.trained_model)

# Setup filepaths
pl_ext.model_type = estimator #before, this will be "not_trained" -> This is needed as long as ext val objects do not get a true model_type assigned 
pl_ext.pipeline_output_path = path
pl_ext.user_input.target_to_validate_on = "status"

#Apply one-hot encoder
pl_ext.ohe.transform(pl_ext.data.X_val) 

# Create master_rfc
pl_ext.build_master_RFC()

#Initialize eval class
pl_ext.evaluation(only_val=True)

#Export evaluation
pl_ext.save_values_for_validation()


#Save Pipeline object for future reference
pl_ext.save_Pipeline()
