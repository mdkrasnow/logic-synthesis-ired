## What we’re trying to achieve

### The problem in IRED terms

On the continuous matrix tasks, IRED learns an **energy landscape over outputs**. During sampling / refinement, the model follows gradients (directly or implicitly through denoising) toward **low-energy regions**. If the landscape contains **spurious low-energy basins** (wrong solutions that are attractive), inference can converge to incorrect outputs even when the correct output exists and is “close.”

IRED already tries to mitigate this by:

* **mining a hard negative** via inner-loop optimization (`opt_step`) and
* using an NCE-style energy loss to enforce `E(real) < E(fake)` at training time. 

But that negative is mined from a very particular initialization distribution (a noised ground-truth sample). That means you can still end up with “blind spots” where the model learns to defend against some wrong basins but not the ones it will hit under OOD structure or realistic failure modes.

### The goal of “logic carving”

You want to **actively carve out families of wrong basins** by generating **structured, hypothesis-driven wrong solutions** (logic negatives) that represent plausible failure modes for the task (sign mistakes, transpose confusions, regularized inverses, wrong-rank completions, trivial fills, etc.). Then you:

1. optionally **snap those hypotheses into the model’s actual preferred basins** using the existing `opt_step` miner, and
2. train the energy model so those wrong basins are **high energy** relative to the true solution.

In one sentence:

> We are reshaping the learned energy landscape so that gradient-based refinement is far less likely to converge to incorrect local minima, especially under distribution shifts, by explicitly training against structured counterexamples.

---

## Key metrics for success (what to measure and how)

### Primary metric: Test-time MSE (ID)

This is already computed by `Trainer1D.evaluate()` as `mean((sample - label)^2)`. 
**Success:** your method yields *lower* test MSE than baseline IRED on a fixed test set.

**Important:** because the datasets are stochastic per `__getitem__`, you must evaluate on a **fixed cached test set** to make comparisons meaningful (your earlier plan). Otherwise you’re measuring dataset RNG variance, not model improvement.

### Secondary metric: OOD Test MSE

The datasets have explicit OOD behavior by scaling dimensions (e.g., `Inverse(..., ood=True)` changes matrix sizes for test). 
**Success:** your method improves (or degrades less) on OOD MSE vs baseline. This is the whole point of “carving” for generalization.

### Landscape metric: Energy gap / margin

Track something like:

* `ΔE = E(real) - min_k E(neg_k)` for your negative set.

In the existing supervise-energy branch, energy is already computed for real vs fake and used in cross-entropy on `-energy_stack`. 
**Success:** over training, the gap becomes more negative (meaning real energy lower than negatives by a larger margin), *without* harming MSE.

### Hard negative failure rate (debugging metric)

Count how often your logic negatives are:

* **too easy** (very high energy already; they don’t shape anything), or
* **too hard / incorrect** (they accidentally match the ground truth; label noise), or
* **collapse** to the same negative every time (generator artifacts).

A simple diagnostic:

* average MSE between each logic negative and ground truth, plus diversity stats (pairwise distances among negatives).

### Stability / convergence metric (sampling health)

If carving works, inference should become more stable:

* fewer sample runs that “blow up” (NaNs),
* fewer cases where sample refinement stagnates in high-error minima.

This matters because you’re adding stronger contrastive forces; it can destabilize training if misweighted.

---

## What “good” looks like after implementation

### In-distribution:

* Lower test MSE (even modestly, e.g., 5–15%) at the same compute budget.
* Similar or slightly better variance across random seeds.

### Out-of-distribution:

* Clear improvement in OOD test MSE relative to baseline.
* Particularly on tasks where plausible wrong attractors exist (inverse is a good one).

### Mechanistically:

* The model should assign **higher energy** to structured wrong hypotheses than baseline does.
* Inner-loop mining starting from those hypotheses should be less able to “find” a low-energy wrong basin.

---

## Common issues to avoid

### 1) Unfair evaluation (the #1 pitfall)

Because matrix datasets sample randomly every time, comparing two runs without caching test data is misleading. You need:

* fixed `FiniteWrapper` with a seed + saved `.npz`
* use `split='test'` for evaluation datasets instead of reusing train. (Right now, matrix tasks set `validation_dataset = dataset` which is the train dataset object.) 

### 2) Label noise from wrong “logic negatives”

If your “negative” occasionally equals the correct output (or becomes correct after `opt_step`), you introduce contradictory supervision: you’re telling the model to push up energy on true solutions.

For each generated logic negative:

* check `mse(neg, gt)` above a threshold before including it.

### 3) Negatives that are too trivial (no learning signal)

If your negatives are so obviously wrong that they already have high energy, they won’t affect gradients. This is why using `opt_step` “snap-to-basin” is valuable: it finds the low-energy representative of that hypothesis class.

### 4) Training instability from over-weighted energy loss

The energy term can dominate and break denoising quality. You’ll see:

* MSE plateaus or worsens
* energy loss collapses to near-zero but task error stays bad
* sometimes NaNs from aggressive inner-loop steps

Mitigation:

* start with small `logic_neg_weight` (e.g., 0.1–0.5)
* ramp it up over epochs
* keep `K` small at first (1–4)

### 5) “Generator artifact learning”

If the logic generator has consistent superficial patterns (e.g., always negating the output, always same perturbation), the model may learn to detect those artifacts rather than the intended logical structure. Keep the generator diverse:

* randomize coefficients (small scalings, epsilons)
* randomize which operator is used
* mix multiple families of hypotheses

### 6) Compute blow-up

Each added negative can require:

* extra `q_sample`,
* extra `opt_step`,
* extra energy forward passes

Start with:

* `K=2` and no opt_step snapping, then add snapping only for 1 negative, then scale.

---

## Things to double-check (implementation correctness checklist)

### Dataset semantics (matrix task decoding)

Your logic negative generator must correctly interpret how `inp` and `out` are packed for each dataset:

* Addition’s input is concatenated flattened `R_one` and `R_two`; output is flattened `R_one + R_two`. 
* Inverse’s input includes flattened matrices and output is concatenated inverses. 
* LowRankCompletion’s input is masked entries; output is full matrix. 

If you decode wrong shapes, you’ll silently generate garbage negatives and training will “work” but not improve anything.

### Multi-negative CE shape logic

Currently, the energy reshape assumes exactly 2 replicas:
`rearrange(energy, '(b r) -> b r', b=b)`. 
When you generalize to `r = 2 + K`, confirm:

* `inp_cat` and `img_cat` are stacked in exactly the same order
* `t_cat` repeats correctly (`t.repeat(r)`)
* target labels are correct (real class = 0)

### Consistency of diffusion time

Your logic negatives must be compared at the same timestep `t`:

* generate `x_neg` in x0-space
* convert to `x_neg_t = q_sample(x_neg, t, noise)` like other uses of noising. 

### Avoid leaking gradients through mining if you don’t intend to

If you use `opt_step` to snap hypotheses to basins, you generally want it to act as a **mining operator**, not an unrolled differentiable path (unless you explicitly choose that). Ensure you’re not accidentally retaining computation graphs for large unrolled loops.

### Checkpoints + eval mode

For fair baseline vs variant:

* same evaluation set cache
* same number of diffusion steps
* same seed handling

---

## Interpreting outcomes (what results mean)

### If ID MSE improves but OOD doesn’t

You probably generated negatives that reflect common in-distribution mistakes, but not the right OOD structure. Improve the hypothesis library (e.g., for inverse: use regularization, diagonal-only, pseudo-inverse approximations).

### If OOD MSE improves but ID slightly worsens

That can be acceptable depending on your goal (generalization). But usually you can recover ID by lowering loss weight or improving negative correctness checks.

### If both worsen

Common causes:

* energy loss weight too high
* negatives sometimes equal truth (label noise)
* test set not fixed (evaluation noise)
* shape decode bug in generator

---

## Practical success criteria for the first matrix-only milestone

For a first “ship-it” result (addition → inverse → lowrank):

* **ID test MSE:** ≥ 5% improvement on at least one task, no regressions > 2% on others
* **OOD test MSE:** ≥ 10% improvement on inverse or lowrank
* **Stability:** no increase in NaNs; training curves remain smooth
* **Energy gap:** consistent increase in separation between true and logic negatives

---

Below is a concrete, code-grounded plan to add **logic-guided synthesis / hard-negative carving** into **IRED** for the **continuous matrix tasks only**, and to measure **test-time MSE vs baseline IRED**.

---

## 0) What the codebase already gives us (so we reuse it)

### Training already has an “energy landscape supervision” slot + inner-loop negative mining

In `GaussianDiffusion1D.p_losses`, IRED already creates a **hard negative** by:

1. taking the current noised sample `data_sample`,
2. running inner-loop optimization `opt_step(...)`,
3. converting the optimized result back to an x0-like candidate (`xmin_noise_rescale`),
4. training an NCE-style energy loss contrasting **real** vs **fake** energies. 

This is exactly where we inject “logic-based synthesis” as **additional structured negatives** (not replacing IRED’s existing inner-loop negatives—augmenting them).

### Evaluation already reports MSE for matrix tasks

`Trainer1D.evaluate` computes `mse_error = (all_samples - label).pow(2).mean()` using `ema_model.sample(...)`. 

And `train.py` sets `metric = 'mse'` for matrix datasets and passes it into `Trainer1D(...)`. 

So: **we don’t need to invent an MSE evaluator**. We *do* need to make the evaluation **deterministic and truly “test split”** to compare baseline vs your variant fairly (details below).

---

## 1) First fix: deterministic “test MSE” for matrix tasks (baseline vs your method)

### Problem

The matrix datasets generate samples using fresh randomness every `__getitem__` call (e.g., `np.random.normal`, `np.random.binomial`). Example: `Addition.__getitem__` draws fresh matrices each time. 
So baseline and your method will be evaluated on **different test draws**, making MSE comparisons noisy/unreliable.

### The code already has a “finite dataset caching” utility, but it assumes 3 returns

`FiniteWrapper.finitize(...)` currently expects `inp, out, trace = dataset[i]` and saves `traces`. 
Matrix tasks return only `(inp, out)` (e.g., `return R_corrupt.reshape(-1), R.reshape(-1)` for addition). 

### Change set A (small, surgical): make `FiniteWrapper` support 2-tuples + optional seed

**File:** `data/dataset.py`
**Edits:**

1. In `FiniteWrapper.finitize`, accept either 2 or 3 values:

   * if `len(item)==2`: set `trace=None`
2. Add `seed` arg to `FiniteWrapper.__init__`, and in `finitize()` do `np.random.seed(seed)` before generating samples so the cached `.npz` is reproducible.

This turns FiniteWrapper into a **canonical, reusable fixed test set** for matrix tasks.

### Change set B: actually use “test” split in `train.py` for validation

Currently, matrix tasks set `validation_dataset = dataset` where `dataset` is created as `'train'` (same object). 
But each dataset class supports `split='test'` and also has an OOD scaling mode:

* `Inverse`: test+OOD uses larger scaling. 
* `Addition`: test+OOD uses larger scaling. 
* `LowRankDataset`: test+OOD uses larger scaling. 

**File:** `train.py`
**Edits:**

* Keep `train_dataset = <Task>('train', ...)`
* Set `val_id = <Task>('test', rank, ood=False)`
* Set `val_ood = <Task>('test', rank, ood=True)` (optional, but very nice to track)
* Wrap validation datasets with `FiniteWrapper(..., n=FLAGS.eval_n, seed=FLAGS.eval_seed)` so they’re fixed across runs.

This gives you **stable ID-test MSE** and optionally **stable OOD-test MSE**.

---

## 2) The new feature: “logic-carving” structured negatives for matrix tasks

### Goal (operational)

Augment IRED’s existing NCE energy loss (real vs inner-loop negative) to become:

**Real vs {inner-loop hard negative + K logic-synthesized negatives}**

This is directly aligned with:

* Noise-Contrastive Estimation as a classification problem ([Journal of Machine Learning Research][1])
* Contrastive divergence / improved CD in EBMs ([U of T Computer Science][2])
* Joint EBM + diffusion sampler training ideas (diffusion as trainable sampler) ([arXiv][3])
* Hard negative sampling literature (controllable “hardness”) ([arXiv][4])

### Where to implement in code

All of the energy contrastive logic for continuous tasks is already centralized in `GaussianDiffusion1D.p_losses(...)`. 
That’s where we extend it.

---

## 3) Concrete code changes to implement logic-carving

### 3.1 Add CLI flags and plumb them into the diffusion model

**File:** `train.py`

Add flags:

* `--logic_carving` (bool)
* `--logic_neg_k` (int, e.g., 4)
* `--logic_neg_weight` (float, e.g., 1.0)
* `--logic_neg_mix_p` (float, optional: probability to include logic negatives on a batch)
* `--eval_only` (bool)
* `--eval_seed`, `--eval_n` (for fixed test set)

When constructing `GaussianDiffusion1D(...)`, pass these through (similar to how `supervise_energy_landscape`, `use_innerloop_opt`, `continuous`, etc. are already passed). 

**File:** `diffusion_lib/denoising_diffusion_pytorch_1d.py`
Update `GaussianDiffusion1D.__init__` signature to store:

* `self.logic_carving`
* `self.logic_neg_k`
* `self.logic_neg_weight`
* `self.task_name` (e.g., `FLAGS.dataset`) so we know which matrix logic to apply

(You can follow the existing pattern where `continuous`, `sudoku`, `baseline`, etc. are stored in init. )

---

### 3.2 Implement task-specific “logic negative generators” (matrix-only)

Create a new file, e.g.:

**File:** `diffusion_lib/logic_carving.py`

Define a simple interface:

```python
def generate_matrix_logic_negatives(task_name, inp, x_true, k, eps):
    # inp: (B, D_in) flattened
    # x_true: (B, D_out) flattened (true output)
    # returns list[tensor] each (B, D_out)
```

Then implement per task using the dataset semantics:

#### Addition task

Dataset builds input as concatenation of two flattened matrices and output as their sum:

* `R_corrupt = concat(R_one.flatten(), R_two.flatten())`
* `out = (R_one + R_two).flatten()` 

So you can decode:

* `R_one, R_two` from `inp`
* compute plausible “wrong hypotheses”:

  * sign mistake: `R_one - R_two`
  * swap+sign: `R_two - R_one`
  * negated sum: `-(R_one + R_two)`
  * scale drift: `α*(R_one + R_two)` with α≈0.8–1.2
  * localized perturbation: add small low-rank noise to mimic “nearly right but wrong”

These are *logic-structured* because they correspond to common algebraic failure modes, not random junk.

#### Inverse task

Dataset builds `A = A0 @ B0`, output `inv(A)` (flattened). 
Negatives:

* transpose confusions: `A.T`
* diagonal-only inverse: `diag(1/diag(A))`
* regularized inverse: `inv(A + λI)` (λ small)
* pseudo-inverse with wrong damping

#### Low-rank completion

Dataset samples `R=A@B`, masks entries, input is `R*mask` flattened, output is full `R` flattened. 
Negatives:

* “fill missing with 0” completion (the trivial hypothesis)
* “fill missing with row/col means” completion
* wrong-rank low-rank projection (e.g., rank-1 approximation when rank=r)

Even if mask isn’t returned, you can approximate it as `mask_hat = (inp_reshaped != 0)` (works well when values are continuous).

---

### 3.3 Extend the existing energy NCE loss to multi-negative (1 + K)

Right now, `p_losses` builds a 2-way classification:

* `energy_stack = cat([energy_real, energy_fake], dim=-1)`
* `loss_energy = cross_entropy(-energy_stack, label=0)` 

We change it to:

* `energy_stack = cat([energy_real] + [energy_fake_innerloop] + energy_fake_logic_list, dim=-1)`
* target class still `0` (the real sample)

**Implementation detail (very important):**
Energies are computed by concatenating batches and reshaping. The current code assumes exactly 2 replicas via:
`energy = rearrange(energy, '(b r) -> b r', b=b)` 
You’ll generalize to `r = 1 + 1 + K` (real + innerloop + K logic):

* Build `img_cat = torch.cat([data_cond, xmin_noise_rescale] + logic_neg_xt_list, dim=0)`
* `inp_cat = inp.repeat(r, 1)` (as done now, but generalized) 
* reshape energy with `r` instead of hardcoded 2

#### How to put logic negatives into the same “time t” space

`p_losses` compares energies at time `t` using `t_cat = t.repeat(2)` today. 
For each logic negative in x0-space (`x_neg`), map it to an xt-space sample at the same t via `q_sample(x_start=x_neg, t=t, noise=randn)` (same mechanism used elsewhere, e.g. conditioning). 

So: logic negatives become *diffusion-consistent* negatives for the energy classifier.

#### Loss weighting

Total loss currently returns `(loss_mse + loss_opt + loss_energy)` (logged as separate pieces in Trainer). 
You’ll keep that structure but set:

* `loss_energy_total = loss_energy_innerloop + logic_neg_weight * loss_energy_logic`
  or unify them into one `(1+K+1)`-way CE.

---

### 3.4 Optional: use logic negatives to **initialize** inner-loop optimization

During training, IRED makes a hard negative by optimizing from `data_sample` (an xt). 
You can make that sharper by sometimes initializing `img` at a logic negative xt instead of `data_sample`, then running `opt_step(...)` (same API). The optimizer is simple gradient descent on the energy:
`img = img - sf * lr * grad` 

This turns “logic hypotheses” into *attractor probes*—exactly your “carve the wrong minima” framing.

Start conservative:

* 50% batches: default behavior
* 50% batches: init from a random logic negative

---

## 4) Do we need new testing tools?

### We already have MSE computation, checkpointing, loading

* `Trainer1D.load(...)` loads checkpoints 
* `Trainer1D.evaluate(...)` prints MSE 

### But we *do* need deterministic evaluation sets (see §1)

That’s the key missing “tooling” for clean baseline-vs-variant MSE comparisons.

### Add “eval-only” mode (small quality-of-life improvement)

Right now, `FLAGS.evaluate` only runs one evaluation and then trains. 
Add `--eval_only` so you can do:

* Train baseline → save checkpoint
* Train logic-carving → save checkpoint
* Run `--eval_only --load_milestone <ckpt>` on the same cached `FiniteWrapper` test set

---

## 5) Minimal experiment plan (matrix tasks only)

1. **Implement deterministic test set**

   * Add `FiniteWrapper` support for 2-tuples + seed. 
   * Modify `train.py` matrix tasks to use `split='test'` validation datasets (ID + optional OOD) using the dataset’s built-in scale shift rules. 

2. **Baseline run**

   * `dataset=addition` (start simplest), fixed `eval_seed`, fixed `eval_n`
   * record test MSE

3. **Logic-carving run**

   * same hyperparams, but `--logic_carving --logic_neg_k=4 --logic_neg_weight=1.0`
   * record test MSE

4. **Ablations (quick)**

   * K ∈ {1, 2, 4, 8}
   * with/without using logic negatives as opt_step init
   * track both ID-test and OOD-test MSE

---

## 6) Why this should reduce “wrong local minima” in IRED terms

IRED’s current negative mining (`opt_step`) is powerful, but it will preferentially find **whatever low-energy basins are easiest to reach** from the current xt. Your proposal adds *targeted basins* (logic hypotheses) that represent **structured, semantically plausible wrong answers**, forcing the energy model to separate:

* the true solution basin
* the “almost-correct but logically wrong” basins

This is directly analogous to why hard-negative mining helps contrastive objectives, and why EBMs benefit from better negative sampling distributions (NCE/CD lineage). ([Journal of Machine Learning Research][1])

---

If you want, I can turn this into a **file-by-file patch checklist** (exact function signatures + pseudo-code blocks matching the current `p_losses` control flow) for:

* `data/dataset.py` (FiniteWrapper fixes)
* `train.py` (test split + deterministic caching + eval_only)
* `diffusion_lib/denoising_diffusion_pytorch_1d.py` (multi-negative NCE)
* `diffusion_lib/logic_carving.py` (matrix hypothesis generators)

[1]: https://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf?utm_source=chatgpt.com "Noise-Contrastive Estimation of Unnormalized Statistical ..."
[2]: https://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf?utm_source=chatgpt.com "On Contrastive Divergence Learning"
[3]: https://arxiv.org/abs/2312.03397?utm_source=chatgpt.com "Joint Training of Energy-Based Model and Diffusion ..."
[4]: https://arxiv.org/abs/2010.04592?utm_source=chatgpt.com "Contrastive Learning with Hard Negative Samples"
