# HRTF upsampling with a generative adversarial network using a gnomonic equiangular projection noise processing extensions

This code is an extension of the code found here: https://github.com/ahogg/HRTF-upsampling-with-a-generative-adversarial-network-using-a-gnomonic-equiangular-projection

The original code base was part of the following research paper:

A. O. T. Hogg, M. Jenkins, H. Liu, I. Squires, S. J. Cooper and L. Picinali: HRTF upsampling with a generative adversarial network using a gnomonic equiangular projection. *In: Proc. IEEE/ACM Transactions on Audio Speech and Language Processing (submitted)*.

Additions to the code have been marked with ****....Added Code*... and ****...End Added Code*...

The original files that have been modified are the following: 

/model/dataset.py
/model/util.py
/config.py
main.py
/evaluation/evaluation.py


Additionally, the repository contains: 
Two Jupyter Notebooks used in Google Colab for noise generation and graph creation
CSV files with results 
perceptual_evaluation.py - An extension of the run_localisation_evaluation function in evaluation.py that allows the  processing of the results of multiple experiments with one run
Example job submission files used on the HPC
