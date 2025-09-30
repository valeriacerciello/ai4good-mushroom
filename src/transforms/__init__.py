import sys
import copy
import torchvision.transforms as tvt
from omegaconf import OmegaConf


# Fuse all transforms defined in this project with the torchvision
# transforms. Special attention is given to local transforms that may
# have the same name as some torchvision transform
_local_tr = sys.modules[__name__]
_tvt_tr = sys.modules["torchvision.transforms"]

_intersection_tr = set(_local_tr.__dict__) & set(_tvt_tr.__dict__)
_intersection_tr = set([t for t in _intersection_tr if not t.startswith("_")])
_intersection_cls = []

for name in _intersection_tr:
    cls = getattr(_local_tr, name)
    if not "torchvision.transforms." in str(cls):
        _intersection_cls.append(cls)

if len(_intersection_tr) > 0:
    if len(_intersection_cls) > 0:
        raise Exception(
            f"It seems that you are overriding a transform from "
            f"torchvision, this is forbidden, please rename your classes "
            f"{_intersection_tr} from {_intersection_cls}")
    else:
        raise Exception(
            f"It seems you are importing transforms {_intersection_tr} "
            f"from torchvision within the current code base. Please, "
            f"remove them or add them within a class, function, etc.")


def instantiate_transform(transform_option, attr="transform"):
    """Create a transform from an OmegaConf dict such as:

    ```yaml
    transform: CenterCrop
        params:
            size: 10
    ```
    """
    # Read the transform name
    tr_name = getattr(transform_option, attr, None)

    # Find the transform class corresponding to the name
    cls = getattr(_local_tr, tr_name, None)
    if not cls:
        cls = getattr(_tvt_tr, tr_name, None)
        if not cls:
            raise ValueError(f"Transform {tr_name} is nowhere to be found")

    # Parse the transform arguments
    try:
        tr_params = transform_option.get('params')  # Update to OmegaConf 2.0
        if tr_params is not None:
            tr_params = OmegaConf.to_container(tr_params, resolve=True)
    except KeyError:
        tr_params = None
    try:
        lparams = transform_option.get('lparams')  # Update to OmegaConf 2.0
        if lparams is not None:
            lparams = OmegaConf.to_container(lparams, resolve=True)
    except KeyError:
        lparams = None

    # Instantiate the transform
    if tr_params and lparams:
        return cls(*lparams, **tr_params)
    if tr_params:
        return cls(**tr_params)
    if lparams:
        return cls(*lparams)
    return cls()


def instantiate_transforms(transform_options):
    """Create a composite transform from an OmegaConf list such as:

    ```yaml
    - transform: CenterCrop
        params:
            size: 10
    - transform: Normalize
        params:
            mean: 0.5
            std: 0.7
    ```
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))

    if len(transforms) <= 1:
        return tvt.Compose(transforms)

    return tvt.Compose(transforms)


def instantiate_datamodule_transforms(transform_options, log=None):
    """Create a dictionary of composite transforms from a datamodule
    OmegaConf holding lists of transforms characterized by a
    `*transform*` key such as:

    ```yaml
    # parsed in the output dictionary
    train_transform:
        - transform: CenterCrop
            params:
                size: 10
        - transform: Normalize
            params:
                mean: 0.5
                std: 0.7

    # not parsed in the output dictionary
    foo:
        a: 1
        b: 10

    # parsed in the output dictionary
    val_transform:
        - transform: ColorJitter
        - transform: GaussianBlur
    ```

    This helper function is typically intended for instantiating the
    transforms of a `DataModule` from an Omegaconf config object

    Credit: https://github.com/torch-points3d/torch-points3d
    """
    transforms_dict = {}
    for key_name in transform_options.keys():
        if "transform" not in key_name:
            continue
        name = key_name.replace("transforms", "transform")
        params = getattr(transform_options, key_name, None)
        if params is None:
            continue
        try:
            transform = instantiate_transforms(params)
        except Exception:
            msg = f"Error trying to create {name}, {params}"
            log.exception(msg) if log is not None else print(msg)
            continue
        transforms_dict[name] = transform
    if len(transforms_dict) == 0:
        msg = (f"Could not find any '*transform*' key among the provided config"
               f" keys: {transform_options.keys()}. Are you sure you passed a "
               f"datamodule config as input ?")
        log.exception(msg) if log is not None else print(msg)
    return transforms_dict


def explode_transform(transform_list):
    """Extracts a flattened list of transforms from a Compose or from a
    list of transforms.
    """
    out = []
    if transform_list is not None:
        if isinstance(transform_list, tvt.Compose):
            out = copy.deepcopy(transform_list.transforms)
        elif isinstance(transform_list, list):
            out = copy.deepcopy(transform_list)
        else:
            raise Exception(
                "Transforms should be provided either within a list or "
                "a Compose")
    return out
