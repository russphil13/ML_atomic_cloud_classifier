from custom.spe import analyze_and_write, get_analysis_results
from custom.spe import get_clouds, get_ROI_params
from custom.spe import write_experiment_parameters, write_image_data
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
from pathlib import Path

path_main = Path('/home/bob/development/experiment_data/2018')
date_list = ['05-02','05-06']
dir_ext = 'filtered5615'
path_image_folders = [path_main.joinpath(dir_date, dir_ext)
                      for dir_date in date_list]
params_file = path_main.joinpath('cloudparams_s20.csv')
pattern = 'tof=5.*\.spe'

write_experiment_parameters(image_files)
write_image_data(image_files)
analyze_and_write(image_files, params_file)

_, roiL, roiR = get_ROI_params(params_file)

results_list = get_analysis_results(conn, schema='atomic_cloud_images')
#close_connection(conn)

results = np.array(results_list)
plt.plot(results[:,-2], results[:,-1], 'k.')
plt.show()

images_list = get_clouds(roiL)
plt.imshow(images_list[100][1])
plt.show()
