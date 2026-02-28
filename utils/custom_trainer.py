"""
Custom Trainer: 扩展 TensorBoard 记录（所有验证指标）、按参数规范保存路径与模型文件名。
"""
import os
import shutil
from recbole.trainer import Trainer
from recbole.utils import set_color, dict2str, early_stopping
from time import time


class TCFMambaTrainer(Trainer):
    """TCFMamba 定制 Trainer：TB 记录 monitoring_metrics、saved_model_file/checkpoint 按参数命名。"""

    def __init__(self, config, model):
        super().__init__(config, model)
        fcd = config.final_config_dict
        self._monitoring_metrics = fcd.get("monitoring_metrics") or []
        if "saved_model_file" in fcd and fcd["saved_model_file"]:
            self.saved_model_file = fcd["saved_model_file"]
        if fcd.get("log_tensorboard") and fcd.get("tensorboard_dir"):
            from torch.utils.tensorboard import SummaryWriter
            # 父类已用 get_tensorboard(logger) 创建了带日期/随机名的目录，关闭并删掉，只保留 run_name 目录
            if getattr(self.tensorboard, "log_dir", None) and os.path.isdir(self.tensorboard.log_dir):
                self.tensorboard.close()
                try:
                    shutil.rmtree(self.tensorboard.log_dir, ignore_errors=True)
                except Exception:
                    pass
            self.tensorboard = SummaryWriter(fcd["tensorboard_dir"])

    def _add_valid_metrics_to_tensorboard(self, epoch_idx, valid_result):
        """将 valid_result 中指定或全部指标写入 TensorBoard。"""
        if not self._monitoring_metrics:
            for k, v in valid_result.items():
                if isinstance(v, (int, float)):
                    self.tensorboard.add_scalar("valid/%s" % k, v, epoch_idx)
        else:
            for k in self._monitoring_metrics:
                if k in valid_result and isinstance(valid_result[k], (int, float)):
                    self.tensorboard.add_scalar("valid/%s" % k, valid_result[k], epoch_idx)

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
                self._add_valid_metrics_to_tensorboard(epoch_idx, valid_result)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
