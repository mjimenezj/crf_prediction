"""
ETL utilities for NHANES datasets.
"""

import os
import warnings
from datetime import datetime

import pandas as pd


#=============================================================================#

def xpt_to_df(folder_path):
    '''
    Opens NHANES .xpt files (SAS format) as Pandas DataFrames 
    and returns a {df_names: dfs} dictionary.
    '''
    # Dict for storing DataFrames
    dataframes = {}

    # Iterating over .xpt files
    for filename in os.listdir(folder_path):
        if filename.endswith(".xpt"):
            # Base name 
            base_name = os.path.splitext(filename)[0]
        
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
        
            # Read the .xpt file
            df = pd.read_sas(file_path, format="xport")
        
            # Store the DataFrame using the base name
            dataframes[base_name] = df
            print(f"Loaded: {base_name} with shape {df.shape}")
        else:
            pass

    print(f'{len(dataframes.keys())}.xpt files have been opened as Pandas DataFrames')

    return dataframes


#=============================================================================#

def extract_selected_variables(dataframes: dict,
                               variables_by_dataset: dict,
                               cohorts: list = None) -> dict:
    """
    Extracts specific NHANES variables from cohort datasets (e.g., DEMO_A, HIQ_B), 
    including participant ID (SEQN) and survey cycle (SDDSRVYR), and 
    concatenates them into one DataFrame per source file (e.g., DEMO, HIQ, etc.).

    Parameters:
        dataframes (dict): Dictionary with keys like 'DEMO_A', 'HIQ_B', etc.,
                           each containing a Pandas DataFrame.
        variables_by_dataset (dict): Dictionary mapping dataset base names (e.g. "DEMO", "SMQ")
                                     to the list of required variables. 
                                     e.g.: {"DEMO": ["SEQN", "SDDSRVYR", "RIDAGEYR"],
                                             "HIQ": ["SEQN", "HID010"],...}
        cohorts (list, optional): List of cohort suffixes (e.g., ["A","B","C"]).
                                  Defaults to ["A","B","C"].
                                  NNHANES suffixes: Cohort A = 1999-2000, 
                                                           B = 2001-2002, ...

    Returns:
        dict: Dictionary of concatenated and filtered DataFrames by dataset type 
              (e.g., {"DEMO": df, "SMQ": df, ...})
    """
    if cohorts is None:
        cohorts = ["A", "B", "C"]

    final_dfs = {}

    for base_name, variables in variables_by_dataset.items():
        cohort_dfs = []
        missing_files = []
        for cohort in cohorts:
            df_key = f"{base_name}_{cohort}"
            if df_key not in dataframes:
                missing_files.append(df_key)
                continue

            df = dataframes[df_key]
            selected_columns = [col for col in variables if col in df.columns]

            if not selected_columns:
                warnings.warn(
                    f"No matching columns found in dataset '{df_key}'. "
                    f"Expected at least one of: {variables}"
                )
                continue

            try:
                df_subset = df[selected_columns].copy()
                cohort_dfs.append(df_subset)
            except Exception as e:
                warnings.warn(f"Error processing dataset '{df_key}': {str(e)}")

        if missing_files:
            warnings.warn(
                f"Missing datasets for {base_name}: {', '.join(missing_files)}"
            )

        if cohort_dfs:
            try:
                # Concatenate cohort_dfs in a single df per type.
                final_dfs[base_name] = pd.concat(cohort_dfs, ignore_index=True)
            except Exception as e:
                warnings.warn(
                    f"Error concatenating dataframes for {base_name}: {str(e)}"
                )
        else:
            warnings.warn(f"No valid dataframes to concatenate for {base_name}.")
    
    if final_dfs:
        print(
            f"Variables were selected successfully and {len(dataframes)} dataframes "
            f"corresponding to {len(cohorts)} cohorts were concatenated to "
            f"{len(final_dfs)} dataframes."
        )
    else:
        print("No valid datasets were processed.")
        
    return final_dfs


#=============================================================================#

def validate_concatenation(dataframes: dict, concatenated_dfs: dict):
    """
    Measure the shape of individual cohort DataFrames (A, B, C) and validate the
    concatenated DataFrame. Outputs are printed to the console.
    """

    # Log header
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("NHANES DATA VALIDATION REPORT")
    print(f"Generated on: {timestamp}")
    print("This log validates that the concatenated DataFrames contain the correct number of rows")
    print("based on their respective A, B, and C cohort files.\n")

    # Validation for each base dataset
    for base_name, concatenated_df in concatenated_dfs.items():
        cohorts = ['A', 'B', 'C']
        row_counts = []
        total_rows = 0

        print("==============================================")
        print(f"\tSource: {base_name}")

        for cohort in cohorts:
            key = f"{base_name}_{cohort}"
            if key in dataframes:
                shape = dataframes[key].shape
                print(f"{key} shape = {shape}")
                row_counts.append(shape[0])
                total_rows += shape[0]

        row_sum_str = " + ".join(str(n) for n in row_counts)
        print(f"{row_sum_str} = {total_rows}\n")

        concat_shape = concatenated_df.shape
        print(f"Concatenated {base_name} shape = {concat_shape}")

        match_status = "MATCH" if concat_shape[0] == total_rows else "MISMATCH"
        print(f"{match_status}: {total_rows} = {concat_shape[0]}")

    print("==============================================")



#=============================================================================#

def merge_datasets(concatenated_dfs: dict) -> pd.DataFrame:
    """
    JOIN all datasets in the final_dfs dictionary ON 'SEQN' to form a single
    unified DataFrame.
    
    Parameters:
        final_dfs (dict): Dictionary of concatenated DataFrames by dataset 
        type (e.g., {'DEMO': df, 'SMQ': df, ...}).
    
    Returns:
        pd.DataFrame: Merged DataFrame containing all datasets, merged on 'SEQN'.
    """
    if not concatenated_dfs:
        warnings.warn("No valid datasets to merge.")
        return None

    # Start with the first dataset and merge others one by one
    merged_df = next(iter(concatenated_dfs.values()))
    print(f"Merging started with \t{merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")
    
    for key, df in concatenated_dfs.items():
        if key != "DEMO":  # Skip merging DEMO again, as it is already in merged_df
            merged_df = pd.merge(merged_df, df, on="SEQN", how="outer")
            print(f"Merged {key}: \t\t{merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")

    # Final log after merging all datasets
    print(f"Final merged dataframe shape: {merged_df.shape}")
    
    return merged_df

#=============================================================================#

def descriptors(df):
    """
    Returns descriptors of a given dataframe: count, mean, std, min, 
    Q25%, Q50%, Q75%, max, number of unique values and number of 
    null values.
    """
    descriptors = df.describe().transpose()
    descriptors['unique_values'] = df.nunique()
    descriptors['null_values'] = df.isnull().sum()
    
    return descriptors


#=============================================================================#

