import multiprocessing as mp
import os
from functools import partial

import datasets
import hydra
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import tqdm
import transformers
import wandb
from torch.utils.data import DataLoader

from lmrax.datasets.preference_feedback import FlaxDataCollatorForSeq2SeqPF
from lmrax.datasets.utils import seed_worker, shp_to_pf


def predict_fn(params, batch, model, reward_id, rng=None):
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
    ).logits[..., reward_id]

    rejected_reward = model.decode(
        params=params,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=context["attention_mask"],
        decoder_input_ids=rejected["input_ids"],
        decoder_attention_mask=rejected["attention_mask"],
        train=training,
        dropout_rng=rejected_rng,
    ).logits[..., reward_id]

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


def loss_fn(params, batch, dropout_rng, model, reward_id):
    weight = batch["weight"]
    log_prob_chosen, log_prob_rejected = predict_fn(
        params, batch, model, reward_id, dropout_rng
    )
    loss = -jnp.mean(
        weight * log_prob_chosen + (1 - weight) * log_prob_rejected
    )

    return loss


def grad_fn(params, batch, rng, model, reward_id):
    return jax.value_and_grad(loss_fn)(params, batch, rng, model, reward_id)


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(0, 1, 6),
    in_axes=(None, None, 0, 0, 0, 0, None),
)
def update_fn(model, optimizer, rng, batch, params, state, reward_id):
    _, rng = jax.random.split(rng)
    loss, grads = grad_fn(params, batch, rng, model, reward_id)
    grads = jax.lax.pmean(grads, "batch")
    updates, state = optimizer.update(grads, state, params)
    grad_norm = grad_norm_fn(updates)
    params = optax.apply_updates(params, updates)
    return loss, params, state, grad_norm, rng


@partial(jax.jit, static_argnums=(1,))
def batch_sharding(batch, n):
    return jax.tree_map(
        lambda x: x.reshape(n, x.shape[0] // n, *x.shape[1:]), batch
    )


@partial(jax.jit, static_argnums=(1,))
def batch_select(batch, idx):
    return jax.tree_map(lambda x: x[idx], batch)


@partial(jax.pmap, static_broadcasted_argnums=(2, 3))
def eval_fn(params, batch, model, reward_id):
    weight = batch["weight"]
    log_prob_chosen, log_prob_rejected = predict_fn(
        params, batch, model, reward_id
    )
    loss = -jnp.mean(
        weight * log_prob_chosen + (1 - weight) * log_prob_rejected
    )
    acc = jnp.mean(log_prob_chosen > log_prob_rejected)
    return loss, acc


@jax.jit
def mean_grads_fn(grads):
    return jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)


@jax.jit
def grad_norm_fn(grads):
    grads, _ = jax.flatten_util.ravel_pytree(grads)
    return jnp.linalg.norm(grads)


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
        self.reward_id = self.tokenizer.vocab[cfg.reward_token]
        self.batch_size = cfg.batch_size_per_device * jax.device_count()
        self.max_length = cfg.max_length

        self.train_loader = self.get_dataloader(self.train_ds)
        self.val_loader = self.get_dataloader(
            self.val_ds, shuffle=False, max_length=self.cfg.max_length
        )

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
                batch = batch_sharding(batch, jax.device_count())
                loss, params, state, grad_norm, rng = update_fn(
                    self.model,
                    self.optimizer,
                    rng,
                    batch,
                    params,
                    state,
                    self.reward_id,
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

    def train(self, params, state, rng):
        for i in range(self.cfg.epochs):
            self.epoch += 1
            params, state, rng = self.train_epoch(params, state, rng)

    def evaluate(self, params):
        avg_loss = 0.0
        avg_acc = 0.0
        with tqdm.tqdm(self.val_loader, desc="Evaluating") as bar:
            for i, batch in enumerate(bar):
                batch = batch_sharding(batch, jax.device_count())
                loss, acc = eval_fn(params, batch, self.model, self.reward_id)
                loss = jax.device_get(loss)
                acc = jax.device_get(acc)

                avg_loss += loss.mean()
                avg_acc += acc.mean()

                bar.set_postfix({"loss": loss.mean(), "acc": acc.mean()})

        avg_loss /= len(self.val_loader)
        avg_acc /= len(self.val_loader)

        return {"loss": avg_loss, "acc": avg_acc}


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg):

    train_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.train)
    train_ds = train_ds.filter(
        lambda x: x["score_ratio"] >= cfg.dataset.score_ratio,
        num_proc=mp.cpu_count(),
    )
    train_ds = train_ds.map(
        shp_to_pf,
        remove_columns=train_ds.features.keys(),
        num_proc=mp.cpu_count(),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)

    val_ds = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.val)
    val_ds = val_ds.map(
        shp_to_pf,
        remove_columns=val_ds.features.keys(),
        num_proc=mp.cpu_count(),
    )

    rng = jax.random.PRNGKey(cfg.seed)
    model, params = transformers.FlaxAutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        _do_init=False,
    )
    optimizer = optax.chain(
        optax.clip(cfg.max_grad_value),
        optax.adamw(
            cfg.optimizer.lr,
            b1=cfg.optimizer.b1,
            b2=cfg.optimizer.b2,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        ),
    )

    optimizer = optax.MultiSteps(optimizer, cfg.gradient_accumulation)
    params = model.init_weights(rng, (1, 1), params)
    state = optimizer.init(params)

    params = jax.device_put_replicated(params, jax.devices())
    state = jax.device_put_replicated(state, jax.devices())
    rng = jax.device_put_sharded(
        list(jax.random.split(rng, jax.device_count())), jax.devices()
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        optimizer=optimizer,
    )
    wandb.init(project="pf", config=cfg, dir=cfg.save_dir)
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.config["reward_id"] = trainer.reward_id
    wandb.run.config["train_size"] = len(train_ds)
    wandb.run.config["val_size"] = len(val_ds)

    trainer.train(params, state, rng)
    wandb.finish()


if __name__ == "__main__":
    main()
