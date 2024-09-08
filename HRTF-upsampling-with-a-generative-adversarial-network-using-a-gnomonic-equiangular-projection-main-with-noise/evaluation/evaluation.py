from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
from preprocessing.utils import convert_to_sofa

import shutil
from pathlib import Path

import glob
import torch
import pickle
import os
import re
import numpy as np

import matlab.engine

def replace_nodes(config, sr_dir, file_name):
    # Overwrite the generated points that exist in the original data
    with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
        hr_hrtf = pickle.load(f)

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)

    lr_hrtf = torch.permute(
        downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor),
        (1, 2, 3, 0))

    lr = lr_hrtf.detach().cpu()
    for p in range(5):
        for w in range(config.hrtf_size):
            for h in range(config.hrtf_size):
                if hr_hrtf[p, w, h] in lr:
                    sr_hrtf[p, w, h] = hr_hrtf[p, w, h]

    generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
    target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

    return target, generated

def run_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):

    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
            # Overwrite the generated points that exist in the original data
            with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{sr_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        lsd_errors = []
        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)
            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))

#*********************Added Code to Return the Mean LSD Error for the Personalization Evaluation************************************************************************    
    mean_lsd_error = np.mean([error[1] for error in lsd_errors])
    print('Mean LSD Error: %0.3f' % mean_lsd_error)
    with open(f'{config.path}/{file_ext}', "wb") as file:
        pickle.dump(lsd_errors, file)
    
    return mean_lsd_error

#*********************End of Added Code to Return the mean LSD error for the Personalization Evaluation************************************************************************ 


#*********************Added Code to Perform Personalization Evaluation************************************************************************ 
def lsd_personalization_evaluation(config, sr_dir, file_ext=None):
    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
    lsd_errors = []

    # Load all high-res HRTFs
    all_hrtf_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
    all_hrtf_file_names = ['/' + os.path.basename(x) for x in all_hrtf_paths]

    sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
    sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

    for file_name in sr_data_file_names:
        target, generated = replace_nodes(config, sr_dir, file_name)
        subject_id = ''.join(re.findall(r'\d+', file_name))

        subject_lsd_errors = []
        for hr_file_name in all_hrtf_file_names:
            hr_subject_id = ''.join(re.findall(r'\d+', hr_file_name))
            if hr_subject_id != subject_id:
                with open(config.valid_hrtf_merge_dir + hr_file_name, "rb") as f:
                    hr_hrtf = pickle.load(f)
                #permutation that would have been applied in the replace_nodes function
                hr_target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))
                error = spectral_distortion_metric(generated, hr_target)
                subject_lsd_errors.append(float(error.detach()))
                print('LSD Error of subject %s comparing to %s: %0.4f' % (subject_id, hr_subject_id, float(error.detach())))

        mean_error = np.mean(subject_lsd_errors)
        std_dev_error = np.std(subject_lsd_errors)
        lsd_errors.append([subject_id, mean_error, std_dev_error])
        print('Mean LSD Error for subject %s: %0.3f, Std Dev: %0.3f' % (subject_id, mean_error, std_dev_error))

    overall_mean_error = np.mean([error[1] for error in lsd_errors])
    overall_std_dev_error = np.std([error[1] for error in lsd_errors])
    print('Overall Mean LSD Error: %0.3f, Overall Std Dev: %0.3f' % (overall_mean_error, overall_std_dev_error))
    
    # Compare with the mean LSD from run_lsd_evaluation, modifcation
    mean_lsd_evaluation_error = run_lsd_evaluation(config, sr_dir)
    discrepancy = overall_mean_error - mean_lsd_evaluation_error
    discrepancy_within_std_dev = abs(discrepancy) <= overall_std_dev_error
    print('Discrepancy between overall mean LSD and run_lsd_evaluation mean LSD: %0.3f' % discrepancy)
    print('Is the discrepancy within the standard deviation? %s' % discrepancy_within_std_dev)

    with open(f'{config.path}/{file_ext}', "wb") as file:
        pickle.dump({'lsd_errors': lsd_errors, 'overall_mean_error': overall_mean_error, 'overall_std_dev_error': overall_std_dev_error, 'mean_lsd_evaluation_error': mean_lsd_evaluation_error, 'discrepancy': discrepancy, 'discrepancy_within_std_dev': discrepancy_within_std_dev}, file)
#*********************End of Added Code to Perform Personalization Evaluation************************************************************************ 

def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        nodes_replaced_path = sr_dir
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_hrtf_merge_dir + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file)

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        convert_to_sofa(nodes_replaced_path, config, cube, sphere)
        print('Created valid sofa files')

        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path + '/sofa_min_phase')]

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    # Added the evaluation folder to the MATLAB path so that the function is visible
    eng.addpath(os.path.abspath("evaluation"), nargout=0) 

    loc_errors = []
    for file in hrtf_file_names:
        target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
        if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
            generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
        else:
            generated_sofa_file = nodes_replaced_path+'/sofa_min_phase/' + file

        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{config.path}/{file_ext}', "wb") as file:
        pickle.dump(loc_errors, file)


def run_target_localisation_evaluation(config):

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    # Added the evaluation folder to the MATLAB path so that the function is visible
    eng.addpath(os.path.abspath("evaluation"), nargout=0)       

    loc_target_errors = []
    target_sofa_path = config.valid_hrtf_merge_dir + '/sofa_min_phase'
    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(target_sofa_path)]
    for file in hrtf_file_names:
        target_sofa_file = target_sofa_path + '/' + file
        generated_sofa_file = target_sofa_file
        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_target_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_target_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_target_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_target_errors]))
    with open(f'{config.data_dir}/{config.dataset}_loc_target_valid_errors.pickle', "wb") as file:
        pickle.dump(loc_target_errors, file)
