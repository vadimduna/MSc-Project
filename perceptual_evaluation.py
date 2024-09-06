import matlab.engine
import os
import re
import numpy as np
import pickle
"""
#*********This code is an extended version of the run_localisation_evaluation function in evaluation.py that allows the
  processing of the results of multiple experiments with one run*************************
"""
def run_target_localization_evaluation(amt_dir, evaluation_root_dir, target_sofa_root_dir, data_dir, matlab_func_path, file_ext):
    eng = matlab.engine.start_matlab()
    s = eng.genpath(amt_dir)
    eng.addpath(s, nargout=0)
    
    # Added the evaluation folder to the MATLAB path so that the function is visible
    eng.addpath(matlab_func_path, nargout=0) 

    # Loop through evaluation folders 
    for evaluation_folder in os.listdir(evaluation_root_dir):
        eval_folder_path = os.path.join(evaluation_root_dir, evaluation_folder)
        nodes_replaced_path = os.path.join(eval_folder_path)
        if not os.path.isdir(nodes_replaced_path):
            continue

        # Loop through experiment folders within the evaluation folder
        for experiment_folder in os.listdir(nodes_replaced_path):
            generated_sofa_path = os.path.join(nodes_replaced_path, experiment_folder)
            if not os.path.isdir(generated_sofa_path):
                continue

            target_sofa_path = os.path.join(target_sofa_root_dir, 'sofa_min_phase_target')
            
            loc_errors = []

            if not os.path.exists(target_sofa_path):
                print(f"Target SOFA path does not exist: {target_sofa_path}")
                continue

            target_hrtf_files = [os.path.join(target_sofa_path, f) for f in os.listdir(target_sofa_path) if f.endswith('.sofa')]

            for target_sofa_file in target_hrtf_files:
                file_name = os.path.basename(target_sofa_file)
                subject_id = ''.join(re.findall(r'\d+', file_name))
                generated_sofa_file = os.path.join(generated_sofa_path, file_name)

                print(f'Target: {target_sofa_file}')
                print(f'Generated: {generated_sofa_file}')

                try:
                    print(f"About to run test for {subject_id} in {eval_folder_path}\n")
                    [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
                    loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
                    print(f'pol_acc1: {pol_acc1}')
                    print(f'pol_rms1: {pol_rms1}')
                    print(f'querr1: {querr1}')
                except Exception as e:
                    print(f"Error processing {generated_sofa_file}: {e}")
                    loc_errors.append([subject_id, float('nan'), float('nan'), float('nan')])

            # Save loc_errors as a pickle file
            pickle_file_name = f"{eval_folder_path}_{file_ext}"
            pickle_file_path = os.path.join(data_dir, pickle_file_name)
            with open(pickle_file_path, "wb") as pickle_file:
                pickle.dump(loc_errors, pickle_file)

            # Write results to a text file for this experiment
            text_file_name = f"{eval_folder_path}_results.txt"
            text_file_path = os.path.join(data_dir, text_file_name)
            with open(text_file_path, "w") as text_file:
                text_file.write(f'Mean ACC Error: %0.3f\n' % np.nanmean([error[1] for error in loc_errors]))
                text_file.write(f'Mean RMS Error: %0.3f\n' % np.nanmean([error[2] for error in loc_errors]))
                text_file.write(f'Mean QUERR Error: %0.3f\n\n' % np.nanmean([error[3] for error in loc_errors]))
                
                text_file.write("Individual Errors:\n")
                for error in loc_errors:
                    text_file.write(f'Subject ID: {error[0]}\n')
                    text_file.write(f'pol_acc1: {error[1]}\n')
                    text_file.write(f'pol_rms1: {error[2]}\n')
                    text_file.write(f'querr1: {error[3]}\n')
                    text_file.write("-" * 20 + "\n")

def run_target_localization_evaluation_baseline(amt_dir, baseline_sofa_dir, target_sofa_dir, data_dir, matlab_func_path, file_ext):
    eng = matlab.engine.start_matlab()
    s = eng.genpath(amt_dir)
    eng.addpath(s, nargout=0)
    
    # Add the MATLAB function path to the MATLAB path
    eng.addpath(matlab_func_path, nargout=0)

    # List of baseline files to compare
    baseline_files = ['maximum.sofa', 'minimum.sofa']

    # Initialize a list to store localization errors
    loc_errors = []

    # Check if the baseline SOFA directory exists
    if not os.path.exists(baseline_sofa_dir):
        print(f"Baseline SOFA path does not exist: {baseline_sofa_dir}")
        return

    # Check if the target SOFA directory exists
    if not os.path.exists(target_sofa_dir):
        print(f"Target SOFA path does not exist: {target_sofa_dir}")
        return

    # List of target HRTF files
    target_hrtf_files = [os.path.join(target_sofa_dir, f) for f in os.listdir(target_sofa_dir) if f.endswith('.sofa')]
    print(target_hrtf_files)

    # Loop through each baseline file and compare with each target file
    for baseline_file in baseline_files:
        baseline_sofa_file_path = os.path.join(baseline_sofa_dir, baseline_file)

        for target_sofa_file in target_hrtf_files:
            file_name = os.path.basename(target_sofa_file)
            subject_id = ''.join(re.findall(r'\d+', file_name))

            print(f'Target: {target_sofa_file}')
            print(f'Baseline: {baseline_sofa_file_path}')

            try:
                print(f"About to run test for {subject_id} with {baseline_file}\n")
                [pol_acc1, pol_rms1, querr1] = eng.calc_loc(baseline_sofa_file_path, target_sofa_file, nargout=3)
                loc_errors.append([subject_id, baseline_file, pol_acc1, pol_rms1, querr1])
                print(f'pol_acc1: {pol_acc1}')
                print(f'pol_rms1: {pol_rms1}')
                print(f'querr1: {querr1}')
            except Exception as e:
                print(f"Error processing {baseline_sofa_file_path}: {e}")
                loc_errors.append([subject_id, baseline_file, float('nan'), float('nan'), float('nan')])

    # Save loc_errors as a pickle file
    pickle_file_path = os.path.join(data_dir, f'loc_errors_{file_ext}.pickle')
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(loc_errors, pickle_file)

    # Write results to a text file
    text_file_path = os.path.join(data_dir, f"results_{file_ext}.txt")
    with open(text_file_path, "w") as text_file:
        text_file.write(f'Mean ACC Error (maximum.sofa): %0.3f\n' % np.nanmean([error[2] for error in loc_errors if error[1] == 'maximum.sofa']))
        text_file.write(f'Mean RMS Error (maximum.sofa): %0.3f\n' % np.nanmean([error[3] for error in loc_errors if error[1] == 'maximum.sofa']))
        text_file.write(f'Mean QUERR Error (maximum.sofa): %0.3f\n\n' % np.nanmean([error[4] for error in loc_errors if error[1] == 'maximum.sofa']))

        text_file.write(f'Mean ACC Error (minimum.sofa): %0.3f\n' % np.nanmean([error[2] for error in loc_errors if error[1] == 'minimum.sofa']))
        text_file.write(f'Mean RMS Error (minimum.sofa): %0.3f\n' % np.nanmean([error[3] for error in loc_errors if error[1] == 'minimum.sofa']))
        text_file.write(f'Mean QUERR Error (minimum.sofa): %0.3f\n\n' % np.nanmean([error[4] for error in loc_errors if error[1] == 'minimum.sofa']))

        text_file.write("Individual Errors:\n")
        for error in loc_errors:
            text_file.write(f'Subject ID: {error[0]}\n')
            text_file.write(f'Baseline: {error[1]}\n')
            text_file.write(f'pol_acc1: {error[2]}\n')
            text_file.write(f'pol_rms1: {error[3]}\n')
            text_file.write(f'querr1: {error[4]}\n')
            text_file.write("-" * 20 + "\n")

if __name__ == "__main__":
    amt_dir = '/Users/vadimdunaevskiy/Documents/local_HRTF_file/amtoolbox-full-1/amtoolbox-full-1.5.0'  # Path to AMT directory
    evaluation_root_dir = '/Users/vadimdunaevskiy/Documents/local_HRTF_file/perceptual_evaluation_brown_clean_remaining'  # Root path to the evaluation folders
    target_sofa_root_dir = '/Users/vadimdunaevskiy/Documents/local_HRTF_file'  # Path to the root directory containing the target SOFA files
    baseline_sofa_dir = '/Users/vadimdunaevskiy/Documents/local_HRTF_file/sofa_min_phase_baseline'
    data_dir = '/Users/vadimdunaevskiy/Documents/local_HRTF_file/results'  # Path to directory to save results
    matlab_func_path = '/Users/vadimdunaevskiy/Documents/local_HRTF_file/'
    file_ext = 'loc_target_valid_errors.pickle'  # File extension for the result files

    run_target_localization_evaluation(amt_dir, evaluation_root_dir, target_sofa_root_dir, data_dir, matlab_func_path, file_ext)
    #run_target_localization_evaluation_baseline(amt_dir, baseline_sofa_dir, target_sofa_root_dir, data_dir, matlab_func_path, file_ext)
    