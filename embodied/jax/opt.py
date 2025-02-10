import math
from typing import Union, Callable
import re

import jax
import jax.numpy as jnp
import ninjax as nj
import optax
from optax import tree_utils as otu

from . import internal
from . import nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):

  summary_depth: int = 2

  def __init__(self, modules, opt, hessian=False):
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules
    self.opt = opt
    self.hessian = hessian
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(self, lossfn, *args, has_aux=False, **kwargs):
    metrics = {}

    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux
    
    if self.hessian:
      loss, params, grads, hvp, vector, update_precond, aux = grad(
          lossfn2, self.modules, has_aux=True, hessian=True,
          step=self.step.read() + 1)(*args, **kwargs)
      if self.scaling:
        loss *= 1 / self.grad_scale.read()

      counts = {k: math.prod(v.shape) for k, v in params.items()}
      if nj.creating():
        print(self._summarize_params(counts, self.summary_depth))

      axes = internal.get_data_axes()
      if axes:
        grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)
        hvp = jax.tree.map(lambda x: jax.lax.pmean(x, axes), hvp)

      if self.scaling:
        invscale = 1 / self.grad_scale.read()
        grads = jax.tree.map(lambda x: x * invscale, grads)

      state = self.sub('state', nj.Tree, self.opt.init, params)
      updates, new_state = self.opt.update(
        grads, state.read(), params, Hvp=hvp, vector=vector, update_preconditioner=update_precond
      )
    else:
      loss, params, grads, aux = nj.grad(
          lossfn2, self.modules, has_aux=True)(*args, **kwargs)
      if self.scaling:
        loss *= 1 / self.grad_scale.read()

      counts = {k: math.prod(v.shape) for k, v in params.items()}
      if nj.creating():
        print(self._summarize_params(counts, self.summary_depth))

      axes = internal.get_data_axes()
      if axes:
        grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

      if self.scaling:
        invscale = 1 / self.grad_scale.read()
        grads = jax.tree.map(lambda x: x * invscale, grads)

      state = self.sub('state', nj.Tree, self.opt.init, params)
      updates, new_state = self.opt.update(grads, state.read(), params)

    nj.context().update(optax.apply_updates(params, updates))
    state.write(new_state)
    grad_norm = optax.global_norm(grads)
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    metrics['grad_norm'] = grad_norm
    metrics['grad_rms'] = nets.rms(grads)
    metrics['update_rms'] = nets.rms(updates)
    metrics['param_rms'] = nets.rms([x.values for x in self.modules])
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.clip(
            max_prob * jnp.exp(-decay * (n - flat_start)), min_prob, max_prob
        )

    return _schedule


def _prerun(fun, *args, **kwargs):
  if not nj.context().modify and not nj.context().create:
    return set()
  # Copy container structure so modifications inside the user function
  # (e.g. popping from a dict) are not applied during prerun.
  args, kwargs = jax.tree.map(lambda x: x, (args, kwargs))
  state, output, accessed, modified, created = fun(
      dict(nj.context()), *args, ignore=True, track=True,
      seed=nj.seed(None, True), **kwargs)
  del output
  creations = {k: v for k, v in state.items() if k in created}
  nj.context().update(creations)
  return accessed, modified


@jax.named_scope('grad')
def grad(fun, targets, has_aux=False, hessian=False, step=None):
  """Compute the gradient of an impure function with respect to the specified
  state entries or modules. The transformed function returns a tuple containing
  the computed value, selected state entries, their gradients, and if
  applicable auxiliary outputs of the function.
  
  When hessian is True, also computes Hessian vector product for use with PSGD."""
  ctx = nj.context()
  targets = targets if hasattr(targets, '__len__') else (targets,)
  if not has_aux:
    fun = lambda *args, _fun=fun, **kwargs: (_fun(*args, **kwargs), {})
  fun = nj.pure(fun, nested=True)

  def wrapper(*args, **kwargs):
    accessed, modified = _prerun(fun, *args, **kwargs)

    strs = []
    for target in targets:
      if isinstance(target, nj.Module):
        prefix = target.path + '/'
        matches = {k: v for k, v in ctx.items() if k.startswith(prefix)}
      if isinstance(target, str):
        pattern = re.compile(f'^{target}(/.*|$)')
        matches = [k for k in ctx if pattern.match(k)]
      if not matches:
        existing = ', '.join(ctx.keys())
        raise KeyError(
            f"Gradient target '{target}' did not match any state entries. " +
            f'Existing state entries: {existing}')
      strs += matches
    existing = ctx.keys()
    assert all(key in existing for key in strs), (strs, existing)
    x1 = {k: v for k, v in ctx.items() if k in strs}
    x2 = {k: v for k, v in ctx.items() if k not in strs}
    if not x1:
      raise ValueError(
          'No inputs to differentiate with respect to. ' +
          f"User provided targets: '{targets}'")

    for key in x1.keys():
      if key not in accessed:
        raise RuntimeError(
            f"Trying to compute gradient with respect to key '{key}' "
            'but the differentiated function does not access it.\n'
            f'Accessed keys: {list(accessed)}\n'
            f'Gradient keys: {list(strs)}')
    x1 = {k: v for k, v in x1.items() if k in accessed}
    x2 = {k: v for k, v in x2.items() if k in accessed}

    def forward(x1, x2, *args, **kwargs):
      before = {**x1, **x2}
      state, (y, aux) = fun(before, *args, create=False, **kwargs)
      changes = {k: v for k, v in state.items() if k in modified}
      return y, (changes, aux)

    if hessian:
      assert step is not None, 'step is required for hessian computation'
      obj_fn = lambda x1: forward(x1, x2, *args, seed=nj.seed(None, True), **kwargs)
      key = jax.random.fold_in(jax.random.PRNGKey(0), step)
      key1, key2 = jax.random.split(key)

      def grad_fn(x1):
        (y, (changes, aux)), grad = jax.value_and_grad(obj_fn, has_aux=True)(x1)
        return grad, (y, (changes, aux))

      def hvp_fn(x1):
        vector = otu.tree_random_like(key1, x1, jax.random.normal)
        grad, hvp, (y, (changes, aux)) = jax.jvp(grad_fn, (x1,), (vector,), has_aux=True)
        return grad, (y, (changes, aux)), hvp, vector
      
      def g_fn(x1):
        grad, (y, (changes, aux)) = grad_fn(x1)
        dummy_hvp = jax.tree.map(jnp.zeros_like, x1)
        dummy_vector = jax.tree.map(jnp.zeros_like, x1)
        return grad, (y, (changes, aux)), dummy_hvp, dummy_vector

      update_schedule = precond_update_prob_schedule(flat_start=500, min_prob=0.05)
      update_precond = jnp.logical_or(
          jax.random.uniform(key2) < update_schedule(step), step < 2
      )

      dx, (y, (changes, aux)), hvp, vector = jax.lax.cond(update_precond, hvp_fn, g_fn, x1)

      return (
        (y, x1, dx, hvp, vector, update_precond, aux)
        if has_aux
        else (y, x1, dx, hvp, vector, update_precond)
      )
    else:
      backward = jax.value_and_grad(forward, has_aux=True)

      (y, (changes, aux)), dx = backward(
          x1, x2, *args, seed=nj.seed(None, True), **kwargs)
      if ctx.modify:
        ctx.update(changes)
        x1 = {k: ctx[k] for k in x1.keys()}

      return (y, x1, dx, aux) if has_aux else (y, x1, dx)
  return wrapper
