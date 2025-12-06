# Important Notice

The code in this repository is now hosted and maintained by Dynamic Intelligence Lab, a research group at Brown University:

[https://github.com/dynamic-intelligence-lab/parallel_lyapunov_exponents](https://github.com/dynamic-intelligence-lab/parallel_lyapunov_exponents)

Please use that repository. The code here is no longer actively mantained.


# parallel_lyapunov_exponents

Reference implementation of our algorithm for estimating Lyapunov exponents in parallel, via a prefix scan, _orders-of-magnitude faster than with previous methods_, for PyTorch. A quick example is worth a thousand words:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

system = torch.load('lorenz_1M_steps.pt')  # data for 1M steps
jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])
if system['is_continuous']:
    jac_vals = torch.eye(n_dims) + jac_vals * dt  # Euler approximation
jac_vals = jac_vals.to(device=DEVICE)

est_LEs = lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)  # fast!
print('Estimated Lyapunov exponents:', est_LEs.tolist())

est_LLE = lyapunov_exponents.estimate_largest_in_parallel(jac_vals, dt=dt)   # faster!
print('Estimated largest Lyapunov exponent:', est_LLE.item())

print('Compare to true exponents of Lorenz: [0.905, 0.0, -14.572]')
```

Our parallel algorithm leverages [generalized orders of magnitude](https://github.com/glassroom/generalized_orders_of_magnitude) (GOOMs) to be able to handle a larger dynamic range of magnitudes than is possible with floating-point formats, and applies a novel selective-resetting method to prevent deviation states from becoming colinear, as we compute all states in parallel.

## Installing

1. Clone this repository.

2. Install all Python dependencies in `requirements.txt`.

3. There is no third step.


## Using

Import the library with:

```python
import lyapunov_exponents
```

The library provides three public methods:

* `estimate_spectrum_in_parallel`, for estimating the spectrum of Lyapunov exponents, applying the parallel algorithm we propose, incorporating our selective-resetting method, _orders-of-magnitude faster than previous methods_, as described in our paper;

* `estimate_largest_in_parallel`, for estimating only the largest Lyapunov exponent, applying the parallelizable expression we derive in our paper, which is _even faster_ (and, moreover, _it's all you need to determine if a system is chaotic_); and

* `estimate_spectrum_sequentially`, for estimating the spectrum of Lyapunov exponents sequentially, applying the standard method with sequential QR-decompositions. We provide this method as a convenience, so you can benchmark it against the parallel methods.

Below we show how to use all methods to estimate Lyapunov exponents for one dynamical system. Please see each method's docstring for additional details on how to use it.


## Example

We provide a precomputed sequence with one million Jacobian values from the well-known [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) in the file `lorenz_1M_steps.pt`. You can quickly test that the library is working properly by executing the following code:

```python
import torch
import lyapunov_exponents

DEVICE = 'cuda'  # change as needed

# Load sample system data with 1M steps:
system = torch.load('lorenz_1M_steps.pt')
jac_vals, dt, n_dims = (system['jac_vals'], system['dt'], system['n_dims'])

# If necessary, map to Jacobian values of transition func with respect to state:
if system['is_continuous']:
    jac_vals = torch.eye(n_dims) + jac_vals * dt  # Euler approximation

# Move Jacobian values to cuda device for parallel execution:
jac_vals = jac_vals.to(device=DEVICE)

# Estimate the spectrum of Lyapunov exponents in parallel:
est_LEs = lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)
print("Estimated spectrum of Lyapunov exponents for {}:".format(system['name']))
print(est_LEs.tolist())
```

For comparison, the true spectrum of Lorenz is estimated to be `[0.905, 0.0, −14.572]`.

To estimate only the largest Lyapunov exponents in parallel, which is _even faster_, use:

```python
est_LLE = lyapunov_exponents.estimate_largest_in_parallel(jac_vals, dt=dt)
print("Estimated largest Lyapunov exponent for {}:".format(system['name']))
print(est_LLE.item())
```

To estimate the spectrum of Lyapunov exponents sequentially, which is _much slower_, use:

```python
seq_LEs = lyapunov_exponents.estimate_spectrum_sequentially(jac_vals, dt=dt)
print("Sequentially estimated spectrum of Lyapunov exponents for {}:".format(system['name']))
print(seq_LEs.tolist())
```


## Replicating Published Results

We have tested our parallel algorithm on all dynamical systems modeled in [William Gilpin's repository](https://github.com/GilpinLab/dysts), and confirmed that our algorithm estimates the spectrum of Lyapunov exponents in parallel with comparable accuracy to sequential estimation, but with execution times that are _orders of magnitude faster_.

To replicate our benchmarks, install Gilpin's [code](https://github.com/GilpinLab/dysts), compute a sequence of 100,000 Jacobian values for every system modeled, and store all results in a Python list of dictionaries `[dict1, dict2, dict3, ...]` called `systems`, with each dictionary in the list having the following keys for each system:

```python
{
    "name": str,
    "is_continuous": bool,
    "n_dims": int,
    "dt": float,
    "jac_vals": torch.float64,
}
```

The Jacobian values, `"jac_vals"`, should be a tensor with `100,000` x `n_dims` x `n_dims` elements. (The file `lorenz_1M_steps.pt`, in this repository, stores 1M steps for one system with this dictionary format.)

Once you have computed data for all systems and stored it in a Python list of dictionaries `systems = [dict1, dict2, dict3, ...]`, execute the code below to run all benchmarks. IMPORTANT: The code below takes a LONG time to run, because sequential estimation becomes much slower as we increase the number of steps.

```python
import torch
import torch.utils.benchmark
import lyapunov_exponents
from tqdm import tqdm

DEVICE = 'cuda'  # change as needed

benchmarks = []

pbar = tqdm(systems)  # iterator with progress bar
for system in pbar:

    name, dt, n_dims = (system['name'], system['dt'], system['n_dims'])

    jac_vals = system['jac_vals']
    if system['is_continuous']:
        jac_vals = torch.eye(n_dims) + jac_vals * dt  # Euler approximation
    jac_vals = jac_vals.to(dtype=torch.float64, device=DEVICE)

    for n_steps in [10, 100, 1000, 10_000, 100_000]:

        pbar.set_description(f"{name}, {n_steps:,} steps in parallel, 7 runs")
        parallel_mean_time = torch.utils.benchmark.Timer(
            stmt='lyapunov_exponents.estimate_spectrum_in_parallel(jac_vals, dt=dt)',
            setup='from __main__ import lyapunov_exponents',
            globals={'jac_vals': jac_vals[:n_steps], 'dt': dt, }
        ).timeit(7).mean

        pbar.set_description(f"{name}, {n_steps:,} steps sequentially, 7 runs")
        sequential_mean_time = torch.utils.benchmark.Timer(
                stmt='lyapunov_exponents.estimate_spectrum_sequentially(jac_vals, dt=dt)',
                setup='from __main__ import lyapunov_exponents',
                globals={'jac_vals': jac_vals[:n_steps], 'dt': dt, }
            ).timeit(7).mean

        benchmarks.append({
            'System Name': name,
            'Number of Steps': n_steps,
            'Parallel Time (Mean of 7 Runs)': parallel_mean_time,
            'Sequential Time (Mean of 7 Runs)': sequential_mean_time,
        })

torch.save(benchmarks, 'benchmarks.pt')  # load with torch.load('benchmarks.pt')
print(*benchmarks, sep='\n')
```


## Configuring Selective Resetting of Interim Deviation States

Selective resetting is a method we formulate in our paper for conditionally resetting interim states in any linear recurrence (diagonal or not, time-variant or not, over GOOMs or not, over real numbers or other fields) _as we compute all states in the linear recurrence in parallel via a prefix scan_.

Our parallel algorithm for estimating the spectrum of Lyapunov exponents uses selective resetting to prevent deviation states from becoming colinear, as we compute all deviation states in parallel via a prefix scan. The implementation of our parallel algorithm in this repository, `estimate_spectrum_in_parallel()`, accepts two arguments that give you fine-grained control over selective resetting of interim deviation states:

* `max_cos_sim`: a float specifying the maximum cosine similarity allowed between pairs of interim deviation states on any step. Default: 0.99999, _i.e._, selective resetting will be triggered when the cosine similarity of one or more pairs of state vectors exceeds 0.99999.

* `n_above_max`: an integer value specifying the number of pairs of state vectors with cosine similarity above `max_cos_sim` that trigger a selective reset. Default: 1, _i.e._ selective resetting will be triggered if at least one cosine similarity exceeds `max_cos_sim`.


### More Information on Selective Resetting

If you are interested in understanding how our selective-resetting method works, we recommend taking a look at [https://github.com/glassroom/selective_resetting/](https://github.com/glassroom/selective_resetting/), an implementation of selective resetting over real numbers instead of GOOMs. We also recommend reading Appendix C of our paper, which explains the intuition behind selective resetting informally, with step-by-step examples.


## Scaling to Higher-Dimensional Systems


### Largest Lyapunov Exponent of Higher-Dimensional Systems

If your goal is only to determine whether a high-dimensional system is chaotic, you need to estimate only its largest Lyapunov exponent. Our code for parallel estimation of the largest exponent, `estimate_largest_in_parallel`, is both faster and more memory-efficient than the code for estimating all exponents, because it does not require any QR decompositons. It scales better with the number of steps _and_ the number of dimensions, without modification, subject only to the memory limits of a single cuda device.

If execution requires more memory than you have available in a single device, you can pass a custom function that distributes execution of the parallel scan over multiple devices -- _e.g._, by applying parallel scans to different segments of the sequence in different devices, then applying a parallel scan over the partially reduced results in a single device. For example, if your custom parallel scan function is called `MyDistributedReduceScan`, you would execute:

```python
est_LLE = lyapunov_exponents.estimate_largest_in_parallel(
    jac_vals, dt=dt, reduce_scan_func=MyDistributedReduceScan)
```

Your custom `reduce_scan_func` must accept three arguments: (1) a complex tensor with a sequence of matrices of shape `...` x `n_steps` x `n_dims` x `n_dims`, where `...` can be any number of preceding dimensions and `n_steps` may vary, (2) a binary associative function (our code will pass `goom.log_matmul_exp`), and (3) an integer indicating the dimension over which to apply the parallel scan. Your custom `reduce_scan_func` must return a complex tensor of shape `...` x `n_dims` x `n_dims` with the reduced result.


### Spectrum of Lyapunov Exponents of Higher-Dimensional Systems

Our code for parallel estimation of the spectrum of Lyapunov exponents, `estimate_spectrum_in_parallel` relies on a custom QR-decomposition function that scales well with the number of time steps for _low-dimensional_ systems. As the number of dimensions increases, parallel execution of QR-decompositions can saturate a single GPU to near-100% utilization, requiring additional parallel hardware (_e.g._, additional GPUs, additional GPU nodes, a distributed supercomputer) to benefit from parallelization.

If you have access to additional parallel hardware, you can pass a custom QR-decomposition function that takes advantage of such additional hardware. For example, if you have a QR-decomposition function called `MyDistributedQRFunc` that takes advantage of additional parallel hardware, you would execute:

```python
est_LEs = lyapunov_exponents.estimate_spectrum_in_parallel(
    jac_vals, dt=dt, qr_func=MyDistributedQRFunc)
```

Your custom QR-decomposition function must accept a single torch.float64 tensor of shape `...` x `n_dims` x `n_dims`, where `...` can be any number of preceding dimensions, and return a tuple of torch.float64 tensors, _each_ with the same shape (`...` x `n_dims` x `n_dims`), containing, respectively, the $Q$ and $R$ factors for each matrix in the input tensor.

If execution requires more memory than you have available in a single device, you can also pass a custom function that distributes execution of the parallel prefix scan over multiple devices. For example, if your custom parallel prefix scan function is called `MyDistributedPrefixScan`, you would execute:

```python
est_LEs = lyapunov_exponents.estimate_spectrum_in_parallel(
    jac_vals, dt=dt, qr_func=MyDistributedQRFunc, prefix_scan_func=MyDistributedPrefixScan)
```

Your custom `prefix_scan_func` must accept three arguments: (1) a complex tensor with a sequence of matrices of shape `...` x `n_steps` x `n_rows` x `n_cols`, where `...` can be any number of preceding dimensions and `n_steps` may vary, (2) a binary associative function (our code will pass one), and (3) an integer indicating the dimension over which to apply the parallel scan. Your custom `reduce_scan_func` must return a complex tensor of shape `...` x `n_steps` x `n_rows` x `n_cols` with the cumulative matrix states. 


## Possible Further Optimizations

The code in this repository is a reference implementation meant to correspond as closely as possible to the formulation in our paper. We know that further optimizations are possible.

For example, our code in `lyapunov_exponents.py` currently checks if all bias elements at each step on the left are still zeroed with the following statement,

```python
B_on_L_is_still_zeroed = (goom.exp(log_B_on_L) == 0).all(dim=(-2, -1))
```

which could be implemented more efficiently, say, by checking a boolean flag per position.

We have refrained from implementing this and other optimizations because _we want the code here to correspond as closely as possible to the formulation in our paper_. If you want to implement additional optimizations, you're welcome to do so on a separate repository of your own.


## Citing

```
@article{
heinsen2025generalized,
title={Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation},
author={Franz A. Heinsen and Leo Kozachkov},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=SUuzb0SOGu},
note={}
}
```

## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.
