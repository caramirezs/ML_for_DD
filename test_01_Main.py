# ####  ESTE ARCHIVO ESTA EN .gitignore ####git
from lib.main_datasets_fun import uniprot_id_datasets
from lib.grid_XGBoost_fun import BayesSearchCV_XGBoost

from lib.main_func_p1 import timer, path, dir_new
from lib.main_func_p4 import modelXGBoost_fit_scores
from lib.main_func_p4 import brier_score
from lib.main_func_p4 import git_pull, git_push
from datetime import datetime
from collections import OrderedDict

import pandas as pd

# XGBoost library
import xgboost as xgb
import os

#####################################
# proteina (uniprot_ID)
uniprot_id = 'P49841'
metric = 'f1_weighted'
resample_factor = 3
resample_mode = 'under_sampling'
gpu_id = 1
server = 'Test_server'  # Server name

#####################################
# Parametros
seed = 142854
fp_name = 'morgan2_c'
frac_iter = 0.5  # No. jobs  50%
t_max = 2 * 1/60  # Max in hours
path_file = path(uniprot_id)

#####################################
# Fetch & Pull
git_pull()
#####################################

