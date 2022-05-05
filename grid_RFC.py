import multiprocess as mp
import os

from lib.grid_RFC_fun import list_dict_params, param_grid_mod, model_clf_grid_search
import timeit

start = timeit.default_timer()

args = [param_grid_mod(params_dict) for params_dict in list_dict_params]

if __name__ == '__main__':
    with mp.Pool() as pool:
        job = pool.map_async(model_clf_grid_search, args)
        job.get()
stop = timeit.default_timer()
print(f'tiempo transcurrido: {(stop-start)/60} minutos')


# from lib.grid_RFC_fun import list_dict_params, param_grid_mod, model_clf_grid_search
#
# params_dict = {'n_estimators': 80, 'min_samples_split': 8, 'min_samples_leaf': 5, 'max_features': 1.0,
#                'max_leaf_nodes': 100, 'oob_score': False, 'max_samples': None, 'criterion': 'gini'}
#
# resultado = model_clf_grid_search(params_dict)
# print(resultado)

