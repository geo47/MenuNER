import sys
import json


def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def get_tqdm():
    ip = ipython_info()
    if ip == "terminal" or not ip:
        from tqdm import tqdm
        return tqdm
    else:
        try:
            from tqdm import tqdm_notebook
            return tqdm_notebook
        except:
            from tqdm import tqdm
            return tqdm


def read_config(config):
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config


def if_none(origin, other):
    return other if origin is None else origin


def print_model_params(model, print_model=False, print_params=False):

    print("Total Trainable parameters: {}".format(model.get_n_trainable_params()))

    if print_model:
        print(model)

    if print_params:
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))