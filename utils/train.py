"""
Standard training script for TCFMamba and other RecBole models.

Usage:
    # Train TCFMamba on gowalla dataset
    python scripts/train.py --model=TCFMamba --dataset=gowalla --config=config/tcfmamba_gowalla.yaml

    # Train with custom config
    python scripts/train.py --model=TCFMamba --dataset=gowalla \
        --config=config/tcfmamba_gowalla.yaml \
        --learning_rate=0.0005 \
        --epochs=50

    # Resume training from checkpoint
    python scripts/train.py --model=TCFMamba --dataset=gowalla \
        --config=config/tcfmamba_gowalla.yaml \
        --checkpoint=SavedData/TCFMamba-gowalla.pth

Supported models:
    - TCFMamba (this repository)
    - BERT4Rec, GRU4Rec, SRGNN, SINE, etc. (RecBole built-in)
"""

import os
import sys
import argparse
import logging
from logging import getLogger

import torch
from recbole.config import Config
from recbole.trainer import Trainer
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    get_model,
    get_trainer,
)
from recbole.data.transform import construct_transform
from recbole.utils.enum_type import Enum

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TCFMamba
from tcfmamba import TCFMamba


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train recommendation models")

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., TCFMamba, BERT4Rec, GRU4Rec)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., gowalla, foursquare_TKY, foursquare_NYC)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., config/tcfmamba_gowalla.yaml)",
    )

    # Optional arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="GPU ID (overrides config, e.g., '0' or '0,1')",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Evaluation batch size (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saved models and logs",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show training progress",
    )

    return parser.parse_args()


def override_config(config, args):
    """Override config with command line arguments."""
    overrides = {
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value
            print(f"Override config: {key} = {value}")

    # Handle output directory
    if args.output_dir:
        config["checkpoint_dir"] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)

    # Handle logging options
    if args.wandb:
        config["log_wandb"] = True
    if args.tensorboard:
        config["log_tensorboard"] = True

    return config


def train_model(model_name, dataset_name, config_file, args):
    """
    Train a recommendation model.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        config_file: Path to config file
        args: Command line arguments

    Returns:
        dict: Training results
    """
    # Load configuration
    config = Config(model=model_name, dataset=dataset_name, config_file_list=[config_file])
    config = override_config(config, args)

    # Initialize random seed
    init_seed(config["seed"], config["reproducibility"])

    # Initialize logger
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 50)
    logger.info(f"Training {model_name} on {dataset_name}")
    logger.info("=" * 50)
    logger.info(f"Configuration:\n{config}")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset(config)
    logger.info(dataset)

    # Data preparation
    logger.info("Preparing data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Initialize model
    logger.info("Initializing model...")
    init_seed(config["seed"] + config.get("local_rank", 0), config["reproducibility"])

    # Get model class
    if model_name == "TCFMamba":
        model_class = TCFMamba
    else:
        # Load from RecBole
        try:
            model_class = get_model(model_name)
        except ValueError:
            logger.error(f"Model {model_name} not found. Please check the model name.")
            raise

    model = model_class(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # Calculate FLOPs
    transform = construct_transform(config)
    try:
        flops = get_flops(model, dataset, config["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")
    except Exception as e:
        logger.warning(f"Could not calculate FLOPs: {e}")

    # Initialize trainer
    trainer = Trainer(config, model)

    # Resume from checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.resume_checkpoint(args.checkpoint)

    # Training
    logger.info("Starting training...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        verbose=args.verbose,
        saved=True,
        show_progress=args.verbose,
        callback_fn=None,
    )

    # Evaluation
    logger.info("Evaluating on test set...")
    test_result = trainer.evaluate(test_data, show_progress=config.get("show_progress", False))

    # Log environment info
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    # Log results
    logger.info("=" * 50)
    logger.info(set_color("Best Validation Result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test Result", "yellow") + f": {test_result}")
    logger.info("=" * 50)

    # Close W&B logger if used
    if hasattr(trainer, "wandblogger") and trainer.wandblogger:
        try:
            trainer.wandblogger._wandb.finish()
        except:
            pass

    return {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "config": config,
    }


def main():
    """Main entry point."""
    args = get_args()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        results = train_model(
            model_name=args.model,
            dataset_name=args.dataset,
            config_file=args.config,
            args=args,
        )

        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50)
        print(f"Best Validation: {results['best_valid_result']}")
        print(f"Test Result: {results['test_result']}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
