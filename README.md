# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py




testing log on google colab:
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-SamLovesCoding2
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 117 items

tests/test_tensor_general.py ............................................................... [ 53%]
......................................................                                       [100%]


-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================== 117 passed, 4435 warnings in 251.54s (0:04:11) ==========================

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-SamLovesCoding2/minitorch/fast_ops.py (164)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        tensors_misaligned = (                                               |
            len(out_shape) != len(in_shape)                                  |
            or len(out_strides) != len(in_strides)                           |
            or not (out_shape == in_shape).all()-----------------------------| #0
            or not (out_strides == in_strides).all()-------------------------| #1
        )                                                                    |
        if tensors_misaligned:                                               |
            for i in prange(len(out)):---------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        |
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)         |
                to_index(i, out_shape, out_index)                            |
                                                                             |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                                                                             |
                in_position = index_to_position(in_index, in_strides)        |
                out_position = index_to_position(out_index, out_strides)     |
                                                                             |
                out[out_position] = fn(in_storage[in_position])              |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #3, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (180) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (181) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (220)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-SamLovesCoding2/minitorch/fast_ops.py (220)
------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                 |
        out: Storage,                                                         |
        out_shape: Shape,                                                     |
        out_strides: Strides,                                                 |
        a_storage: Storage,                                                   |
        a_shape: Shape,                                                       |
        a_strides: Strides,                                                   |
        b_storage: Storage,                                                   |
        b_shape: Shape,                                                       |
        b_strides: Strides,                                                   |
    ) -> None:                                                                |
        n = len(out)                                                          |
        tensors_misaligned = (                                                |
            len(out_strides) != len(a_strides)                                |
            or len(out_shape) != len(a_shape)                                 |
            or len(a_strides) != len(b_strides)                               |
            or len(b_shape) != len(a_shape)                                   |
            or not (out_strides == a_strides).all()---------------------------| #4
            or not (a_strides == b_strides).all()-----------------------------| #5
            or not (out_shape == a_shape).all()-------------------------------| #6
            or not (a_shape == b_shape).all()---------------------------------| #7
        )                                                                     |
        if tensors_misaligned:                                                |
            # global broadcasting case                                        |
            for i in prange(n):-----------------------------------------------| #9
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                |
                in_index_a = np.empty(MAX_DIMS, dtype=np.int32)               |
                in_index_b = np.empty(MAX_DIMS, dtype=np.int32)               |
                to_index(i, out_shape, out_index)                             |
                                                                              |
                broadcast_index(out_index, out_shape, a_shape, in_index_a)    |
                broadcast_index(out_index, out_shape, b_shape, in_index_b)    |
                                                                              |
                # Calculate ordinals                                          |
                in_position_a = index_to_position(in_index_a, a_strides)      |
                in_position_b = index_to_position(in_index_b, b_strides)      |
                out_position = index_to_position(out_index, out_strides)      |
                out[out_position] = fn(                                       |
                    a_storage[in_position_a], b_storage[in_position_b]        |
                )                                                             |
        else:                                                                 |
            # fast path: aligned tensors                                      |
            for i in prange(n):-----------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #9, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (245) is hoisted out of the
parallel loop labelled #9 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (246) is hoisted out of the
parallel loop labelled #9 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index_a = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (247) is hoisted out of the
parallel loop labelled #9 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index_b = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (289)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-SamLovesCoding2/minitorch/fast_ops.py (289)
------------------------------------------------------------------------|loop #ID
    def _reduce(                                                        |
        out: Storage,                                                   |
        out_shape: Shape,                                               |
        out_strides: Strides,                                           |
        a_storage: Storage,                                             |
        a_shape: Shape,                                                 |
        a_strides: Strides,                                             |
        reduce_dim: int,                                                |
    ) -> None:                                                          |
        for idx in prange(len(out)):------------------------------------| #10
            current_idx: Index = np.empty(MAX_DIMS, dtype=np.int32)     |
            dim_size: int = a_shape[reduce_dim]                         |
            to_index(idx, out_shape, current_idx)                       |
            result_pos = index_to_position(current_idx, out_strides)    |
                                                                        |
            dim_stride = a_strides[reduce_dim]                          |
            start_pos = index_to_position(current_idx, a_strides)       |
            total = out[result_pos]                                     |
            for offset in range(dim_size):                              |
                curr_pos = start_pos + offset * dim_stride              |
                total = fn(total, float(a_storage[curr_pos]))           |
                                                                        |
            out[result_pos] = total                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (299) is hoisted out of the
parallel loop labelled #10 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: current_idx: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/content/mod3-SamLovesCoding2/minitorch/fast_ops.py (316)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-SamLovesCoding2/minitorch/fast_ops.py (316)
---------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                           |
    out: Storage,                                                                      |
    out_shape: Shape,                                                                  |
    out_strides: Strides,                                                              |
    a_storage: Storage,                                                                |
    a_shape: Shape,                                                                    |
    a_strides: Strides,                                                                |
    b_storage: Storage,                                                                |
    b_shape: Shape,                                                                    |
    b_strides: Strides,                                                                |
) -> None:                                                                             |
    """NUMBA tensor matrix multiply function.                                          |
                                                                                       |
    Should work for any tensor shapes that broadcast as long as                        |
                                                                                       |
    ```                                                                                |
    assert a_shape[-1] == b_shape[-2]                                                  |
    ```                                                                                |
                                                                                       |
    Optimizations:                                                                     |
                                                                                       |
    * Outer loop in parallel                                                           |
    * No index buffers or function calls                                               |
    * Inner loop should have no global writes, 1 multiply.                             |
                                                                                       |
                                                                                       |
    Args:                                                                              |
    ----                                                                               |
        out (Storage): storage for `out` tensor                                        |
        out_shape (Shape): shape for `out` tensor                                      |
        out_strides (Strides): strides for `out` tensor                                |
        a_storage (Storage): storage for `a` tensor                                    |
        a_shape (Shape): shape for `a` tensor                                          |
        a_strides (Strides): strides for `a` tensor                                    |
        b_storage (Storage): storage for `b` tensor                                    |
        b_shape (Shape): shape for `b` tensor                                          |
        b_strides (Strides): strides for `b` tensor                                    |
                                                                                       |
    Returns:                                                                           |
    -------                                                                            |
        None : Fills in `out`                                                          |
                                                                                       |
    """                                                                                |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                             |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                             |
                                                                                       |
    for i1 in prange(out_shape[0]):  # batch dimension---------------------------------| #13
        for i2 in prange(out_shape[1]):  # rows----------------------------------------| #12
            for i3 in prange(out_shape[2]):  # columns---------------------------------| #11
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]                      |
                b_inner = i1 * b_batch_stride + i3 * b_strides[2]                      |
                                                                                       |
                acc = 0.0                                                              |
                                                                                       |
                for k in range(a_shape[2]):                                            |
                    acc += (                                                           |
                        a_storage[a_inner + k * a_strides[2]]                          |
                        * b_storage[b_inner + k * b_strides[1]]                        |
                    )                                                                  |
                out_position = (                                                       |
                    i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]    |
                )                                                                      |
                                                                                       |
                out[out_position] = acc                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None







!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Output:
Epoch  0  loss  7.2922255695146845 correct 28
Epoch  10  loss  4.915612246233167 correct 43
Epoch  20  loss  5.035798662073198 correct 45
Epoch  30  loss  4.073779510245649 correct 48
Epoch  40  loss  5.940754087808721 correct 46
Epoch  50  loss  2.5890127543859993 correct 48
Epoch  60  loss  2.8429055305428244 correct 43
Epoch  70  loss  1.5508064680653104 correct 49
Epoch  80  loss  1.749785154876713 correct 50
Epoch  90  loss  0.8964344132519654 correct 49
Epoch  100  loss  0.45658764623099074 correct 49
Epoch  110  loss  0.6048679420629817 correct 49
Epoch  120  loss  0.6511420861257535 correct 49
Epoch  130  loss  0.795121159213063 correct 50
Epoch  140  loss  2.1697621078907963 correct 48
Epoch  150  loss  0.6136973473499121 correct 50
Epoch  160  loss  1.0080943486245653 correct 50
Epoch  170  loss  0.107569387343013 correct 50
Epoch  180  loss  1.9244270264163854 correct 48
Epoch  190  loss  0.41912291604416335 correct 49
Epoch  200  loss  0.19149205185450682 correct 50
Epoch  210  loss  1.1747620896746487 correct 49
Epoch  220  loss  0.2243843175041054 correct 50
Epoch  230  loss  0.1791852744619677 correct 50
Epoch  240  loss  0.8507793358280205 correct 50
Epoch  250  loss  0.5460500191772893 correct 50
Epoch  260  loss  0.86719987190774 correct 50
Epoch  270  loss  0.16156670292809466 correct 49
Epoch  280  loss  1.3277565473880926 correct 50
Epoch  290  loss  0.10218178747288134 correct 50
Epoch  300  loss  0.1309631263364095 correct 50
Epoch  310  loss  1.2655094045446402 correct 50
Epoch  320  loss  0.3328852962717396 correct 50
Epoch  330  loss  1.2352179048171212 correct 50
Epoch  340  loss  0.22035414942734252 correct 49
Epoch  350  loss  0.051244417864177674 correct 50
Epoch  360  loss  0.25672060883274095 correct 49
Epoch  370  loss  0.31037618233155356 correct 49
Epoch  380  loss  0.011889519620869085 correct 50
Epoch  390  loss  0.0932743347628206 correct 49
Epoch  400  loss  0.030058845566652545 correct 50
Epoch  410  loss  0.666420835969128 correct 50
Epoch  420  loss  0.11619804319713775 correct 50
Epoch  430  loss  0.16594170287863563 correct 49
Epoch  440  loss  0.8937706997262358 correct 50
Epoch  450  loss  0.709327504667639 correct 49
Epoch  460  loss  0.5793223289697177 correct 50
Epoch  470  loss  0.3439185813391107 correct 50
Epoch  480  loss  0.03481784181024232 correct 49
Epoch  490  loss  0.09391081666923072 correct 50