
import torch
import os


def _load_mammoth_model(dict_keys, model: torch.nn.Module, args):
    for k in list(dict_keys):
        if args.distributed != 'dp':
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        elif 'module' not in k:
            dict_keys[k.replace('net.', 'net.module.')] = dict_keys.pop(k)

    for k in list(dict_keys):
        if '_features' in dict_keys:
            dict_keys.pop(k)

    if 'lucir' in args.model.lower():
        model.register_buffer('classes_so_far', torch.zeros_like(
            dict_keys['classes_so_far']).to('cpu'))

    model.load_state_dict(dict_keys)
    model.net.to(model.device)
    return model


def _load_net(dict_keys, model: torch.nn.Module, args, ignore_classifier=True):
    for k in list(dict_keys):
        if args.distributed != 'dp':
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        elif 'module' not in k:
            dict_keys[k.replace('net.', 'net.module.')] = dict_keys.pop(k)

    if not ignore_classifier:
        cl_weights = [dict_keys[k] for k in list(dict_keys.keys()) if 'classifier' in k]
        if len(cl_weights) > 0:
            cl_size = cl_weights[-1].shape[0]
            model.net.classifier = torch.nn.Linear(
                model.net.classifier.in_features, cl_size).to(model.device)
    else:
        for k in list(dict_keys):
            if 'classifier' in k:
                dict_keys.pop(k)

    for k in list(dict_keys):
        if '_features' in dict_keys:
            dict_keys.pop(k)
    for k in list(dict_keys):
        if 'net' in k:
            dict_keys[k[4:]] = dict_keys.pop(k)
    for k in list(dict_keys):
        if 'wrappee.' in k:
            dict_keys[k.replace('wrappee.', '')] = dict_keys.pop(k)

    try:
        model.net.load_state_dict(dict_keys)
    except BaseException:
        _, unm = model.net.load_state_dict(dict_keys, strict=False)
        if ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"

    model.net.to(model.device)
    return model


def mammoth_load_checkpoint(args, model: torch.nn.Module, ignore_classifier=False) -> torch.nn.Module:
    """
    Loads the keys from the given checkpoint.
    - Handles DataParallel and DistributedDataParallel checkpoints.
    - Handles checkpoints from previous versions of the code.
    - Handles head initialization for LUCIR.
    :param args: the model with the checkpoint loaded.
    """
    if not os.path.exists(args.loadcheck):
        raise ValueError('The given checkpoint does not exist.')

    saved_obj = torch.load(args.loadcheck, map_location=torch.device("cpu"))

    if 'args' in saved_obj and 'model' in saved_obj:
        _check_loaded_args(args, saved_obj['args'])
        # Mammoth checkpoint
        model = _load_mammoth_model(saved_obj['model'], model, args)
        if 'buffer' in saved_obj:
            loading_model = saved_obj['args'].model
            if args.model != loading_model:
                print(f'WARNING: The loaded model was trained with a different model: {loading_model}')
            model.load_buffer(saved_obj['buffer'])

        return model, saved_obj['results']
    else:
        # Model only checkpoint
        model = _load_net(saved_obj, model, args, ignore_classifier=ignore_classifier)

        return model, None


def _check_loaded_args(args, loaded_args):
    ignored_args = ['loadcheck', 'start_from', 'stop_after', 'conf_jobnum', 'conf_host', 'conf_timestamp', 'distributed', 'examples_log', 'examples_full_log',
                    'intensive_savecheck', 'job_number', 'conf_git_commit', 'loss_log', 'tensorboard', 'seed', 'savecheck', 'notes', 'non_verbose', 'autorelaunch', 'force_compat', 'conf_external_path']
    mismatched_args = [x for x in vars(args) if x not in ignored_args and (
        x not in vars(loaded_args) or getattr(args, x) != getattr(loaded_args, x))]

    if len(mismatched_args):
        if 'force_compat' not in vars(args) or args.force_compat:
            print(
                "WARNING: The following arguments do not match between loaded and current model:")
            print(mismatched_args)
        else:
            raise ValueError(
                'The loaded model was trained with different arguments: {}'.format(mismatched_args))
