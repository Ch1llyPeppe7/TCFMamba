"""
Training script for TCFMamba and RecBole models.
Config: use --config for one file, or omit to use config/dataset/{dataset}.yaml + config/model/{model}.yaml.
Usage:
    python utils/train.py --model=TCFMamba --dataset=gowalla
    python utils/train.py --model=TCFMamba --dataset=gowalla --config=config/custom.yaml
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color, get_model, get_environment
from logging import getLogger

from tcfmamba import TCFMamba
from prepare_datasets import prepare_if_needed


def _clear_dataset_cache(dataset_name, checkpoint_dir="saved"):
    """Remove cached dataset and dataloader pth so next run rebuilds with current config."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved = os.path.join(root, checkpoint_dir)
    if not os.path.isdir(saved):
        return
    removed = []
    for f in os.listdir(saved):
        if f.startswith(dataset_name + "-") and (f.endswith(".pth") or "-dataloader.pth" in f):
            p = os.path.join(saved, f)
            try:
                os.remove(p)
                removed.append(f)
            except OSError:
                pass
    if removed:
        print(f"[CACHE] Cleared {checkpoint_dir}/ for rebuild: {removed}")


def _patch_recbole_numpy_writable():
    """Avoid PyTorch warning: make numpy arrays writable before tensor()."""
    from recbole.data.dataset import dataset
    from recbole.utils import FeatureType
    import torch.nn.utils.rnn as rnn_utils
    from recbole.data.interaction import Interaction

    _orig = dataset.Dataset._dataframe_to_interaction

    def _patched(self, data):
        new_data = {}
        for k in data:
            value = data[k].values
            if hasattr(value, "flags") and not value.flags.writeable:
                value = np.array(value, copy=True)
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                if k in self.config["numerical_features"]:
                    new_data[k] = torch.FloatTensor(value.tolist())
                else:
                    new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(np.array(d, copy=True)[: self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                if k in self.config["numerical_features"]:
                    base = [torch.FloatTensor(np.array(d[0], copy=True)[: self.field2seqlen[k]]) for d in value]
                    base = rnn_utils.pad_sequence(base, batch_first=True)
                    index = [torch.FloatTensor(np.array(d[1], copy=True)[: self.field2seqlen[k]]) for d in value]
                    index = rnn_utils.pad_sequence(index, batch_first=True)
                    new_data[k] = torch.stack([base, index], dim=-1)
                else:
                    seq_data = [torch.FloatTensor(np.array(d, copy=True)[: self.field2seqlen[k]]) for d in value]
                    new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)

    dataset.Dataset._dataframe_to_interaction = _patched


def _config_files(model_name, dataset_name, config_path=None, experiment_name="default"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path:
        p = config_path if os.path.isabs(config_path) else os.path.join(root, config_path)
        return [p] if os.path.isfile(p) else []
    files = []
    exp_yaml = os.path.join(root, "config", "experiment", f"{experiment_name}.yaml")
    if os.path.isfile(exp_yaml):
        files.append(exp_yaml)
    dataset_file = {"gowalla": "gowalla", "foursquare_TKY": "foursquare_tky", "foursquare_NYC": "foursquare_nyc"}.get(dataset_name, dataset_name.lower())
    dataset_yaml = os.path.join(root, "config", "dataset", f"{dataset_file}.yaml")
    if os.path.isfile(dataset_yaml):
        files.append(dataset_yaml)
    if model_name == "TCFMamba":
        model_yaml = os.path.join(root, "config", "model", "tcfmamba.yaml")
        if os.path.isfile(model_yaml):
            files.append(model_yaml)
    return files


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None, help="Single config file (optional)")
    p.add_argument("--experiment", default="default", help="Experiment config name under config/experiment/ (default, debug, quiet)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--gpu_id", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard (overrides experiment config)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--rebuild-dataset", action="store_true", help="Clear cached dataset/dataloaders and rebuild (fix group_by etc.)")
    return p.parse_args()


def train(model_name, dataset_name, config_file_list, args):
    if not prepare_if_needed(dataset_name):
        print(f"[ERROR] Failed to prepare dataset: {dataset_name}")
        sys.exit(1)

    config_dict = {}
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.learning_rate is not None:
        config_dict["learning_rate"] = args.learning_rate
    if args.gpu_id is not None:
        config_dict["gpu_id"] = args.gpu_id
    if args.seed is not None:
        config_dict["seed"] = args.seed
    if args.tensorboard:
        config_dict["log_tensorboard"] = True
    if args.wandb:
        config_dict["log_wandb"] = True

    if model_name == "TCFMamba":
        config = Config(model=TCFMamba, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)
    else:
        config = Config(model=model_name, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)

    # RecBole sequential split requires group_by "user" (not "user_id"); clear cache and fix
    ea = config.final_config_dict.get("eval_args")
    if isinstance(ea, dict):
        gb = ea.get("group_by")
        if gb == "user_id" or (gb not in ("user", None) and str(gb).lower() != "none"):
            ea["group_by"] = "user"
            _clear_dataset_cache(dataset_name, config["checkpoint_dir"] if "checkpoint_dir" in config else "saved")
    if args.rebuild_dataset:
        _clear_dataset_cache(dataset_name, config["checkpoint_dir"] if "checkpoint_dir" in config else "saved")

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info("=" * 50)
    logger.info(f"Training {model_name} on {dataset_name}")
    logger.info("=" * 50)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + (config["local_rank"] if "local_rank" in config else 0), config["reproducibility"])
    if model_name == "TCFMamba":
        model = TCFMamba(config, train_data.dataset).to(config["device"])
    else:
        model = get_model(model_name)(config, train_data.dataset).to(config["device"])
    logger.info(model)

    trainer = Trainer(config, model)
    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.resume_checkpoint(args.checkpoint)

    logger.info("Starting training...")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=True, saved=True, show_progress=True)
    test_result = trainer.evaluate(test_data, show_progress=False)

    logger.info("=" * 50)
    logger.info(set_color("Best Validation", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test Result", "yellow") + f": {test_result}")
    logger.info("=" * 50)
    logger.info("Running environment:\n" + get_environment(config).draw())

    return {"best_valid_score": best_valid_score, "best_valid_result": best_valid_result, "test_result": test_result}


def main():
    args = get_args()
    _patch_recbole_numpy_writable()

    config_file_list = _config_files(args.model, args.dataset, args.config, args.experiment)
    if not config_file_list:
        print("[ERROR] No config file(s) found. Use --config or add config/dataset and config/model yaml.")
        sys.exit(1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        results = train(args.model, args.dataset, config_file_list, args)
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Best Validation: {results['best_valid_result']}")
        print(f"Test Result: {results['test_result']}")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
