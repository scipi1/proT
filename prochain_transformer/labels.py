
from os.path import join

# DIRECTORIES

def get_dirs(root:str):
    DATA_DIR = join(root,"data")
    INPUT_DIR = join(DATA_DIR,"input")
    OUTPUT_DIR = join(DATA_DIR,"output")
    INTERMEDIATE_DIR = join(DATA_DIR,"intermediate")
    EXPERIMENTS_DIR = join(root,"experiments")
    return INPUT_DIR,OUTPUT_DIR,INTERMEDIATE_DIR,EXPERIMENTS_DIR

# TARGET
target_filename = "y_ist.csv"
target_trimmed_filename = "y_trimmed.csv"
target_sep = ","
target_design_label = "SapNummer"
target_version_label = "Version"
target_design_version_label = "SAP_Version"
target_batch_label = "WA"
target_id_label = "id"
target_value_label = "Value"
target_time_label = "CreateDate"
target_pos_label = "Zyklus"


# INPUT
input_design_label = "SAP"
input_version_label = "Version"
input_variable_label = "Variable"
input_process_label = "Process"
input_batch_label = "WA"
input_step_label = "PaPos"
input_id_label = "id"
input_abs_pos_label = "AbsPos"
input_given_label = "Given"
input_value_label = "Value"
input_time_label = "Time"


# INTERMEDIATE FILES
selected_filename = "lookup_selected.xlsx"
lookup_filename = "lookup.xlsx"
input_raw_filename = "x_prochain.csv"
input_leveled_filename = "x_prochain_lev.csv"
booking_check_filename = "booking_check.xlsx"
booking_missing_filename = "booking_missing.csv"
process_missing_filename = "process_missing.csv"
process_lookup_filename = "process_lookup.xlsx"
booking_design_label = "SAP"
booking_version_label = "SAP_Version"
booking_batch_label = "WA"
booking_step_label = "PaPosNumber"
booking_process_raw_label = "Anlage"
booking_process_label = "Process"

# OUTPUT FILES
pred_cross_attention_filename = "cross_att.npy"
pred_output_filename = "output.npy"
pred_target_filename = "target.npy"

# GENERAL
standard_sep = ","