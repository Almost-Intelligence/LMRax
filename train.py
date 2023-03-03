import multiprocessing as mp
import os
from functools import partial

import datasets
import hydra
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import tqdm
import transformers
import wandb
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.pjit import pjit
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import lmrax.optimizers
from lmrax.datasets.preference_feedback import FlaxDataCollatorForSeq2SeqPF
from lmrax.datasets.utils import seed_worker
from lmrax.sharding import get_batch_shardings, get_params_shardings


def predict_fn(params, batch, model, rng=None):
    if rng is None:
        training = False
        encoder_rng, chosen_rng, rejected_rng = None, None, None
    else:
        training = True
        encoder_rng, chosen_rng, rejected_rng = jax.random.split(rng, 3)
    context = batch["context"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    encoder_outputs = model.encode(
        params=params,
        input_ids=context["input_ids"],
        attention_mask=context["attention_mask"],
        train=training,
        dropout_rng=encoder_rng,
    )

    chosen_reward = model.decode(
        params=params,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=context["attention_mask"],
        decoder_input_ids=chosen["input_ids"],
        decoder_attention_mask=chosen["attention_mask"],
        train=training,
        dropout_rng=chosen_rng,
    ).last_hidden_state.mean(axis=-1)

    rejected_reward = model.decode(
        params=params,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=context["attention_mask"],
        decoder_input_ids=rejected["input_ids"],
        decoder_attention_mask=rejected["attention_mask"],
        train=training,
        dropout_rng=rejected_rng,
    ).last_hidden_state.mean(axis=-1)

    # mask out paddings
    chosen_reward = jnp.where(
        chosen["attention_mask"] == 0, 0.0, chosen_reward
    )  # (B, L)

    rejected_reward = jnp.where(
        rejected["attention_mask"] == 0, 0.0, rejected_reward
    )  # (B, L)

    chosen_score = jnp.sum(chosen_reward, axis=-1)  # (B,)
    rejected_score = jnp.sum(rejected_reward, axis=-1)  # (B,)

    log_prob_chosen = jax.nn.log_sigmoid(chosen_score - rejected_score)  # (B,)
    log_prob_rejected = jax.nn.log_sigmoid(
        rejected_score - chosen_score
    )  # (B,)

    return log_prob_chosen, log_prob_rejected


def loss_fn(params, batch, dropout_rng, model):
    weight = batch["weight"]
    log_prob_chosen, log_prob_rejected = predict_fn(
        params, batch, model, dropout_rng
    )
    loss = -jnp.mean(
        weight * log_prob_chosen + (1 - weight) * log_prob_rejected
    )

    return loss


def grad_fn(params, batch, rng, model):
    return jax.value_and_grad(loss_fn)(params, batch, rng, model)


def _update_fn(model, optimizer, rng, batch, params, state):
    _, rng = jax.random.split(rng)
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    loss, grads = grad_fn(params, batch, rng, model)
    grads = jax.tree_map(lambda x: x.astype(jnp.float32), grads)
    grad_norm = grad_norm_fn(grads)
    params = jax.tree_map(lambda x: x.astype(jnp.float32), params)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, state, grad_norm, rng


@partial(jax.jit, static_argnums=(1,))
def batch_select(batch, idx):
    return jax.tree_map(lambda x: x[idx], batch)


def _eval_fn(params, batch, model):
    weight = batch["weight"]
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    log_prob_chosen, log_prob_rejected = predict_fn(params, batch, model)
    loss = -jnp.mean(
        weight * log_prob_chosen + (1 - weight) * log_prob_rejected
    )
    acc = jnp.mean(log_prob_chosen > log_prob_rejected)
    return loss, acc


def mean_grads_fn(grads):
    return jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)


def grad_norm_fn(grads):
    return jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_map(lambda x: jnp.linalg.norm(x) ** 2, grads),
        )
    )


class Trainer:
    def __init__(self, cfg, model, tokenizer, train_ds, val_ds, optimizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.optimizer = optimizer
        self.steps = 0
        self.epoch = 0
        self.params_updates = 0
        self.batch_size = cfg.batch_size_per_device * cfg.num_dp_devices
        self.max_length = cfg.max_length

        self.params_shardings = None
        self.state_shardings = None

        import jax.sharding as shd

        devices = np.array(jax.devices()).reshape(
            cfg.num_dp_devices, cfg.num_tp_devices
        )

        # dp: data parallel, tp: tensor parallel
        self.mesh = shd.Mesh(devices, ("dp", "tp"))

        self.train_loader = self.get_dataloader(self.train_ds)
        self.val_loader = self.get_dataloader(
            self.val_ds, shuffle=False, max_length=self.cfg.max_length
        )

        self.update_fn = None
        self.eval_fn = None
        self.shard_batch_fn = None

    def get_data_collator(self, max_length=None):
        return FlaxDataCollatorForSeq2SeqPF(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=max_length or self.max_length,
            truncation=True,
        )

    def get_dataloader(self, ds, shuffle=True, max_length=None):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=self.get_data_collator(max_length),
            pin_memory=False,
            drop_last=True,
            num_workers=mp.cpu_count(),
            worker_init_fn=seed_worker,
            shuffle=shuffle,
        )

    def update_updates(self):
        self.params_updates = self.steps // self.cfg.gradient_accumulation

    def train_epoch(self, params, state, rng):
        with tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as bar:
            for batch in bar:
                self.steps += 1

                batch = self.shard_batch_fn(batch)
                loss, params, state, grad_norm, rng = self.update_fn(
                    rng, batch, params, state
                )
                self.update_updates()
                post_fix = {
                    "loss": jax.device_get(loss).mean(),
                    "grad_norm": jax.device_get(grad_norm).mean(),
                }
                bar.set_postfix(post_fix)

                if self.steps % self.cfg.gradient_accumulation == 0:
                    wandb.log(
                        {"train/" + k: v for k, v in post_fix.items()},
                        step=self.params_updates,
                    )
                    if self.params_updates % self.cfg.save_steps == 0:
                        os.makedirs(self.cfg.save_dir, exist_ok=True)
                        self.model.save_pretrained(
                            os.path.join(
                                self.cfg.save_dir,
                                f"model_{self.params_updates}",
                            ),
                            params=params,
                            push_to_hub=False,
                        )
                    if self.params_updates % self.cfg.eval_steps == 0:
                        results = self.evaluate(params)
                        wandb.log(
                            {"val/" + k: v for k, v in results.items()},
                            step=self.params_updates,
                        )
                        print(results)

        return params, state, rng

    def init(self, params):
        batch = next(iter(self.train_loader))

        params = jax.tree_map(np.asarray, params)
        params_shardings = freeze(get_params_shardings(self.mesh, params))
        batch_shardings = get_batch_shardings(self.mesh, batch)

        put_params_on_device = pjit(
            lambda x: x,
            out_axis_resources=params_shardings,
        )

        self.shard_batch_fn = pjit(
            lambda x: x,
            out_axis_resources=batch_shardings,
        )

        params = put_params_on_device(params)
        state = self.optimizer.init(params)

        # TODO(yongchanghao): this is a hack
        def get_state_shardings(x):
            x = unfreeze(x)
            if isinstance(x, dict):
                return params_shardings
            return shd.NamedSharding(self.mesh, shd.PartitionSpec())

        state_shardings = jax.tree_util.tree_map(
            get_state_shardings,
            state,
            is_leaf=lambda x: isinstance(
                unfreeze(x), (dict, optax.EmptyState)
            ),
        )

        def wrapped_update_fn(rng, batch, params, state):
            return _update_fn(
                self.model,
                self.optimizer,
                rng,
                batch,
                params,
                state,
            )

        none_shd = shd.NamedSharding(self.mesh, shd.PartitionSpec())

        self.update_fn = pjit(
            wrapped_update_fn,
            in_axis_resources=(
                none_shd,  # rng
                batch_shardings,  # batch
                params_shardings,  # params
                state_shardings,  # state
            ),
            out_axis_resources=(
                none_shd,  # loss
                params_shardings,  # params
                state_shardings,  # state
                none_shd,  # grad_norm
                none_shd,  # rng
            ),
            donate_argnums=(2, 3),
        )

        def wrapped_eval_fn(params, batch):
            return _eval_fn(params, batch, self.model)

        self.eval_fn = pjit(
            wrapped_eval_fn,
            in_axis_resources=(params_shardings, batch_shardings),
            out_axis_resources=(none_shd, none_shd),
        )

        return params, state

    def train(self, params, state, rng):
        for i in range(self.cfg.epochs):
            self.epoch += 1
            params, state, rng = self.train_epoch(params, state, rng)

    def evaluate(self, params):
        avg_loss = 0.0
        avg_acc = 0.0
        with tqdm.tqdm(self.val_loader, desc="Evaluating") as bar:
            for i, batch in enumerate(bar):
                loss, acc = self.eval_fn(params, batch)

                avg_loss += loss.mean()
                avg_acc += acc.mean()

                bar.set_postfix({"loss": loss.mean(), "acc": acc.mean()})

        avg_loss /= len(self.val_loader)
        avg_acc /= len(self.val_loader)

        return {
            "loss": jax.device_get(avg_loss),
            "acc": jax.device_get(avg_acc),
        }


@hydra.main(version_base=None, config_path="config", config_name="tp")
def main(cfg):
    train_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.train)
    ori_train_len = len(train_ds)
    train_ds = train_ds.filter(
        lmrax.datasets.utils.get_filter_fn(cfg),
        num_proc=mp.cpu_count(),
    )
    train_ds = train_ds.map(
        lmrax.datasets.utils.get_map_fn(cfg),
        remove_columns=train_ds.features.keys(),
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)

    val_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.val)
    ori_val_len = len(val_ds)
    val_ds = val_ds.map(
        lmrax.datasets.utils.get_map_fn(cfg),
        remove_columns=val_ds.features.keys(),
        num_proc=mp.cpu_count(),
        load_from_cache_file=False,
    )
    optimizer_cfg = OmegaConf.to_object(cfg.optimizer)
    optimizer_cls = lmrax.optimizers.get(optimizer_cfg.pop("name"))

    optimizer_chains = [
        optimizer_cls(**optimizer_cfg),
    ]
    if cfg.max_grad_norm is not None:
        optimizer_chains.append(optax.clip_by_global_norm(cfg.max_grad_norm))
    elif cfg.max_grad_value is not None:
        optimizer_chains.append(optax.clip(cfg.max_grad_value))
    optimizer = optax.chain(*optimizer_chains)
    optimizer = optax.MultiSteps(optimizer, cfg.gradient_accumulation)

    rng = jax.random.PRNGKey(cfg.seed)
    model, params = transformers.FlaxAutoModel.from_pretrained(
        cfg.model_name,
        _do_init=False,
    )
    rng = jax.tree_map(np.asarray, rng)
    params = jax.tree_map(np.asarray, params)
    params = model.init_weights(rng, (1, 1), params)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer,
    )

    params, state = trainer.init(params)
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_map(lambda x: x.size, params)
    )
    wandb.init(project="pf", config=OmegaConf.to_object(cfg), dir=cfg.save_dir)
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.config["ori_train_size"] = ori_train_len
    wandb.run.config["ori_val_size"] = ori_val_len
    wandb.run.config["train_size"] = len(train_ds)
    wandb.run.config["val_size"] = len(val_ds)
    wandb.run.config["num_params"] = format(num_params, ",")

    trainer.train(params, state, rng)
    wandb.finish()


if __name__ == "__main__":
    main()
