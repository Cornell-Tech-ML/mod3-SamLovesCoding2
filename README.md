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

3.4 demonstration that the GPU is faster than the CPU
Timing summary
Size: 64
    fast: 0.00294
    gpu: 0.00567
Size: 128
    fast: 0.01384
    gpu: 0.01299
Size: 256
    fast: 0.09199
    gpu: 0.04310
Size: 512
    fast: 1.24000
    gpu: 0.22454
Size: 1024
    fast: 8.98118
    gpu: 1.04186

CPU simple
Epoch  0  loss  41.04414988444761 correct 26
Time for last 10 epochs: 3.50 seconds
Epoch  10  loss  23.72655858521437 correct 44
Time for last 10 epochs: 0.01 seconds
Epoch  20  loss  18.751395083841278 correct 45
Time for last 10 epochs: 0.01 seconds
Epoch  30  loss  15.904002387860494 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  40  loss  14.083134362711142 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  50  loss  12.715592802565212 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  60  loss  11.645201539031458 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  70  loss  10.803849617852826 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  80  loss  10.146531352245846 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  90  loss  9.589220233881127 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  100  loss  9.102912094419425 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  110  loss  8.673335260018348 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  120  loss  8.28979268123109 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  130  loss  7.943300494652024 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  140  loss  7.62822179348869 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  150  loss  7.339900556871055 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  160  loss  7.076595549530654 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  170  loss  6.83461457287216 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  180  loss  6.612121035805536 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  190  loss  6.405785800034577 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  200  loss  6.213802976688813 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  210  loss  6.031861885105663 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  220  loss  5.860398191128142 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  230  loss  5.691537820127961 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  240  loss  5.527828915293825 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  250  loss  5.370472481623171 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  260  loss  5.218386025452052 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  270  loss  5.08307620740874 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  280  loss  4.962097338280207 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  290  loss  4.85072812417337 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  300  loss  4.744978441024186 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  310  loss  4.64445874875858 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  320  loss  4.548442935028789 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  330  loss  4.444259384484504 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  340  loss  4.345007467985967 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  350  loss  4.249068282353648 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  360  loss  4.159140857481641 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  370  loss  4.072393673749403 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  380  loss  3.997424919493115 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  390  loss  3.9259601175990744 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  400  loss  3.8570832074368906 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  410  loss  3.7908665913794106 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  420  loss  3.7272052137968235 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  430  loss  3.665913905697836 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  440  loss  3.6069143623278115 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  450  loss  3.5501987034450324 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  460  loss  3.4955409287019164 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  470  loss  3.4434792732685793 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  480  loss  3.3932924067935795 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  490  loss  3.3449664369041567 correct 50
Time for last 10 epochs: 0.01 seconds



Split CPU:
Epoch  0  loss  34.63215082861363 correct 33
Time for last 10 epochs: 3.49 seconds
Epoch  10  loss  27.23770314717664 correct 41
Time for last 10 epochs: 0.01 seconds
Epoch  20  loss  25.01793683461576 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  30  loss  23.84085358014223 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  40  loss  22.97617324945821 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  50  loss  22.296783937590323 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  60  loss  21.650807029750403 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  70  loss  21.038931380781293 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  80  loss  20.455942702111976 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  90  loss  19.879849422921225 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  100  loss  19.321580175731345 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  110  loss  18.788406859915316 correct 43
Time for last 10 epochs: 0.01 seconds
Epoch  120  loss  18.261849624409333 correct 43
Time for last 10 epochs: 0.01 seconds
Epoch  130  loss  17.72789007568216 correct 43
Time for last 10 epochs: 0.01 seconds
Epoch  140  loss  17.195416097594237 correct 44
Time for last 10 epochs: 0.01 seconds
Epoch  150  loss  16.661937807537235 correct 44
Time for last 10 epochs: 0.01 seconds
Epoch  160  loss  16.131320650775013 correct 45
Time for last 10 epochs: 0.01 seconds
Epoch  170  loss  15.628067130777815 correct 45
Time for last 10 epochs: 0.01 seconds
Epoch  180  loss  15.122730122564684 correct 45
Time for last 10 epochs: 0.01 seconds
Epoch  190  loss  14.639824672834907 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  200  loss  14.176398293541878 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  210  loss  13.719801578908985 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  220  loss  13.28468182290931 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  230  loss  12.886065479717336 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  240  loss  12.50255894461434 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  250  loss  12.131160553917855 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  260  loss  11.778257295351967 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  270  loss  11.437585370930018 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  280  loss  11.118372494117096 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  290  loss  10.797509328991003 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  300  loss  10.494604864328723 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  310  loss  10.224439241044381 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  320  loss  9.964973523202575 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  330  loss  9.718835398734454 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  340  loss  9.483757552944846 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  350  loss  9.256675176833701 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  360  loss  9.043877025708973 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  370  loss  8.842077820520094 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  380  loss  8.647345480089118 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  390  loss  8.46125639434842 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  400  loss  8.282220117764837 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  410  loss  8.111633422393918 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  420  loss  7.94788697147383 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  430  loss  7.79082000266985 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  440  loss  7.638375075006856 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  450  loss  7.49270432545792 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  460  loss  7.353112809203163 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  470  loss  7.217033227914592 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  480  loss  7.0868120336795615 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  490  loss  6.965503956725815 correct 48
Time for last 10 epochs: 0.01 seconds

CPU XOR Data:
Epoch  0  loss  33.671985350885414 correct 33
Time for last 10 epochs: 3.42 seconds
Epoch  10  loss  26.210508222082325 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  20  loss  22.187790883801437 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  30  loss  20.030699022227722 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  40  loss  18.53845858678558 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  50  loss  17.359897050628607 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  60  loss  16.38843501664728 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  70  loss  15.546562334609344 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  80  loss  14.772600855732708 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  90  loss  14.071252601821875 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  100  loss  13.442838530148158 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  110  loss  12.893202616137547 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  120  loss  12.384752637366573 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  130  loss  11.909223352172752 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  140  loss  11.468404027124691 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  150  loss  11.069034471590664 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  160  loss  10.696243873605248 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  170  loss  10.346956077277882 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  180  loss  10.021395294198602 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  190  loss  9.716339141450101 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  200  loss  9.42838310901317 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  210  loss  9.155602369102569 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  220  loss  8.896046742288295 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  230  loss  8.650869232906986 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  240  loss  8.419985551090535 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  250  loss  8.201566196021343 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  260  loss  7.99439671612729 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  270  loss  7.796460448908382 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  280  loss  7.607354990593058 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  290  loss  7.423764755120711 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  300  loss  7.24839858844199 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  310  loss  7.0822459260509305 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  320  loss  6.922503473259471 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  330  loss  6.770549029734076 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  340  loss  6.625393647136252 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  350  loss  6.484147659452166 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  360  loss  6.34935697235562 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  370  loss  6.220142069159428 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  380  loss  6.0952780613791715 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  390  loss  5.9757531863969024 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  400  loss  5.86065923476107 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  410  loss  5.750418815312346 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  420  loss  5.647235493136236 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  430  loss  5.54823696172691 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  440  loss  5.4530725016183785 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  450  loss  5.3608667506418985 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  460  loss  5.273369547480237 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  470  loss  5.189430042095501 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  480  loss  5.108555679301156 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  490  loss  5.029798706696703 correct 50
Time for last 10 epochs: 0.01 seconds


Larger CPU XOR data:
Epoch  0  loss  53.94631208852908 correct 25
Time for last 10 epochs: 3.42 seconds
Epoch  10  loss  27.42441751733076 correct 40
Time for last 10 epochs: 0.01 seconds
Epoch  20  loss  21.255728456397076 correct 40
Time for last 10 epochs: 0.01 seconds
Epoch  30  loss  17.87355407953637 correct 41
Time for last 10 epochs: 0.01 seconds
Epoch  40  loss  16.298907524523585 correct 42
Time for last 10 epochs: 0.01 seconds
Epoch  50  loss  15.28238673343798 correct 45
Time for last 10 epochs: 0.01 seconds
Epoch  60  loss  14.463218530550867 correct 46
Time for last 10 epochs: 0.01 seconds
Epoch  70  loss  13.736883905217995 correct 46
Time for last 10 epochs: 0.01 seconds
Epoch  80  loss  13.139503672391717 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  90  loss  12.61463915790725 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  100  loss  12.076610058525892 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  110  loss  11.671721809033976 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  120  loss  11.323228468868233 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  130  loss  11.00335017463719 correct 47
Time for last 10 epochs: 0.01 seconds
Epoch  140  loss  10.715565242722809 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  150  loss  10.446919957958368 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  160  loss  10.179261740722124 correct 48
Time for last 10 epochs: 0.01 seconds
Epoch  170  loss  9.934641073180837 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  180  loss  9.698198292751991 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  190  loss  9.452481956672761 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  200  loss  9.194803636481096 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  210  loss  8.917704151588772 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  220  loss  8.668224403687713 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  230  loss  8.461175593692953 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  240  loss  8.265663366937208 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  250  loss  8.09180917001715 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  260  loss  7.936218796969963 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  270  loss  7.780282295563008 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  280  loss  7.635572143046428 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  290  loss  7.4910184317217325 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  300  loss  7.350866895298213 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  310  loss  7.212910666598537 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  320  loss  7.080867738101855 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  330  loss  6.951405913028307 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  340  loss  6.826701439553862 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  350  loss  6.703450548548544 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  360  loss  6.584392862141258 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  370  loss  6.46707806342033 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  380  loss  6.355156281932807 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  390  loss  6.246019480090332 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  400  loss  6.141022803162943 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  410  loss  6.038200645209319 correct 49
Time for last 10 epochs: 0.01 seconds
Epoch  420  loss  5.939973652562346 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  430  loss  5.840997163890359 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  440  loss  5.744584166732984 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  450  loss  5.6473283607266715 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  460  loss  5.555422087238576 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  470  loss  5.467165671749702 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  480  loss  5.3809089325785004 correct 50
Time for last 10 epochs: 0.01 seconds
Epoch  490  loss  5.294624746790791 correct 50
Time for last 10 epochs: 0.01 seconds

GPU Simple Data:
Epoch  0  loss  88.31632765046872 correct 29
Time for last 10 epochs: 0.56 seconds
Epoch  10  loss  22.380427895336304 correct 40
Time for last 10 epochs: 0.04 seconds
Epoch  20  loss  16.363224412884406 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  30  loss  15.107261737035746 correct 47
Time for last 10 epochs: 0.04 seconds
Epoch  40  loss  11.265250474668681 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  50  loss  9.802173382546226 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  60  loss  7.908716887219963 correct 49
Time for last 10 epochs: 0.04 seconds
Epoch  70  loss  8.451561198921508 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  80  loss  5.432052553353961 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  90  loss  5.922544414974695 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  100  loss  4.872229342819421 correct 50
Time for last 10 epochs: 0.07 seconds
Epoch  110  loss  4.391883670130486 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  120  loss  3.8224614396771175 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  130  loss  3.8537763246704113 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  140  loss  4.118000054422558 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  150  loss  3.2016305069714712 correct 50
Time for last 10 epochs: 0.07 seconds
Epoch  160  loss  2.9963021905595824 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  170  loss  2.920843610452138 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  180  loss  2.5920310599409033 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  190  loss  2.438450582842671 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  200  loss  2.4294190712515222 correct 50
Time for last 10 epochs: 0.07 seconds
Epoch  210  loss  2.377589749883236 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  220  loss  2.0817411007756226 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  230  loss  1.9421722939073054 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  240  loss  1.785263520559531 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  250  loss  1.646900535853376 correct 50
Time for last 10 epochs: 0.06 seconds
Epoch  260  loss  1.947489269870025 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  270  loss  1.9006986380509798 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  280  loss  1.4413595974918465 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  290  loss  1.7010582477975884 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  300  loss  1.6048287354728794 correct 50
Time for last 10 epochs: 0.06 seconds
Epoch  310  loss  1.3193307511775316 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  320  loss  1.4372026720016151 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  330  loss  1.3775237172065067 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  340  loss  1.2379830659580024 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  350  loss  1.2278537725497607 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  360  loss  0.9958083692310776 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  370  loss  1.145682628073065 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  380  loss  0.9942724423004002 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  390  loss  1.0217026044705149 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  400  loss  0.9855273284807118 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  410  loss  0.8434028577609903 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  420  loss  1.1499413351745094 correct 50
Time for last 10 epochs: 0.05 seconds
Epoch  430  loss  0.9664621399123545 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  440  loss  1.0693756129358132 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  450  loss  0.891197268348358 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  460  loss  0.8876371807187984 correct 50
Time for last 10 epochs: 0.06 seconds
Epoch  470  loss  1.0916600578299893 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  480  loss  0.8766713133885808 correct 50
Time for last 10 epochs: 0.04 seconds
Epoch  490  loss  0.7055977646654776 correct 50
Time for last 10 epochs: 0.05 seconds


GPU Split Data:
Epoch  0  loss  141.4294540380916 correct 36
Time for last 10 epochs: 0.44 seconds
Epoch  10  loss  22.960992439208002 correct 43
Time for last 10 epochs: 0.05 seconds
Epoch  20  loss  21.818538954460646 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  30  loss  20.012640605194356 correct 41
Time for last 10 epochs: 0.05 seconds
Epoch  40  loss  18.81840671668788 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  50  loss  19.901822374657236 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  60  loss  17.493090907139155 correct 46
Time for last 10 epochs: 0.06 seconds
Epoch  70  loss  17.137144564287627 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  80  loss  16.131822912325884 correct 45
Time for last 10 epochs: 0.06 seconds
Epoch  90  loss  17.10462844756788 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  100  loss  14.121672828326652 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  110  loss  15.792527171937731 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  120  loss  14.61397845312286 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  130  loss  12.983677996903197 correct 46
Time for last 10 epochs: 0.07 seconds
Epoch  140  loss  14.047256922421944 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  150  loss  12.789692586180312 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  160  loss  11.76112946322225 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  170  loss  11.708283075104028 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  180  loss  11.484503346529293 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  190  loss  11.598735786271572 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  200  loss  11.330083513836477 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  210  loss  10.431417379280992 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  220  loss  10.820067421120907 correct 49
Time for last 10 epochs: 0.06 seconds
Epoch  230  loss  9.831935602496838 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  240  loss  10.705684734730221 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  250  loss  9.8353565343423 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  260  loss  9.913113345690551 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  270  loss  9.714593433284698 correct 45
Time for last 10 epochs: 0.07 seconds
Epoch  280  loss  11.434706966670618 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  290  loss  8.628494460278073 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  300  loss  9.62100588710841 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  310  loss  8.183289789491612 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  320  loss  8.296592922658329 correct 48
Time for last 10 epochs: 0.08 seconds
Epoch  330  loss  7.48856298209052 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  340  loss  7.4277419823397 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  350  loss  7.101774533276503 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  360  loss  7.2595095723644825 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  370  loss  6.956716199007514 correct 48
Time for last 10 epochs: 0.07 seconds
Epoch  380  loss  6.922666846553879 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  390  loss  7.246743313584479 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  400  loss  7.040474081871757 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  410  loss  7.714192036880423 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  420  loss  6.3285577216012 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  430  loss  7.369582417739493 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  440  loss  6.632701050996754 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  450  loss  6.573483582044561 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  460  loss  6.1003091308733906 correct 49
Time for last 10 epochs: 0.07 seconds
Epoch  470  loss  6.357743423751893 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  480  loss  6.073546554812035 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  490  loss  5.842902585964078 correct 49
Time for last 10 epochs: 0.05 seconds


GPU Split Data:
Epoch  0  loss  45.715836940721495 correct 21
Time for last 10 epochs: 0.52 seconds
Epoch  10  loss  31.187745515513576 correct 34
Time for last 10 epochs: 0.04 seconds
Epoch  20  loss  29.019213878159064 correct 33
Time for last 10 epochs: 0.06 seconds
Epoch  30  loss  28.700978442525233 correct 35
Time for last 10 epochs: 0.06 seconds
Epoch  40  loss  28.46892207626908 correct 35
Time for last 10 epochs: 0.05 seconds
Epoch  50  loss  27.495817586674402 correct 37
Time for last 10 epochs: 0.04 seconds
Epoch  60  loss  27.109047862945676 correct 37
Time for last 10 epochs: 0.05 seconds
Epoch  70  loss  26.46789571559659 correct 37
Time for last 10 epochs: 0.06 seconds
Epoch  80  loss  27.3058017520622 correct 37
Time for last 10 epochs: 0.04 seconds
Epoch  90  loss  25.84462030520819 correct 38
Time for last 10 epochs: 0.04 seconds
Epoch  100  loss  25.40935532511636 correct 39
Time for last 10 epochs: 0.04 seconds
Epoch  110  loss  24.328023629872217 correct 39
Time for last 10 epochs: 0.04 seconds
Epoch  120  loss  24.19629699623062 correct 40
Time for last 10 epochs: 0.06 seconds
Epoch  130  loss  23.727201242720362 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  140  loss  23.26874439372304 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  150  loss  24.03924960943573 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  160  loss  22.54815531663298 correct 39
Time for last 10 epochs: 0.04 seconds
Epoch  170  loss  22.247663360176972 correct 41
Time for last 10 epochs: 0.07 seconds
Epoch  180  loss  21.831931013957114 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  190  loss  20.513402959147253 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  200  loss  20.23292919939537 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  210  loss  19.53448264783085 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  220  loss  19.815354919392114 correct 44
Time for last 10 epochs: 0.07 seconds
Epoch  230  loss  18.66682644534995 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  240  loss  18.44044075232521 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  250  loss  19.233777633983546 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  260  loss  16.834344332405962 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  270  loss  18.215072294105596 correct 45
Time for last 10 epochs: 0.07 seconds
Epoch  280  loss  15.619985174234245 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  290  loss  15.521981805095002 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  300  loss  15.158852981265532 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  310  loss  16.434197045987403 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  320  loss  14.973171366730664 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  330  loss  13.808292531424371 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  340  loss  13.410198364961357 correct 47
Time for last 10 epochs: 0.04 seconds
Epoch  350  loss  12.813185209595733 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  360  loss  12.27549842595286 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  370  loss  12.28338226140778 correct 49
Time for last 10 epochs: 0.04 seconds
Epoch  380  loss  11.813869556836384 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  390  loss  11.82311283701993 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  400  loss  11.486994207531595 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  410  loss  11.067900964845645 correct 49
Time for last 10 epochs: 0.04 seconds
Epoch  420  loss  10.172152320270518 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  430  loss  10.59333840906753 correct 49
Time for last 10 epochs: 0.04 seconds
Epoch  440  loss  10.160258933536877 correct 47
Time for last 10 epochs: 0.04 seconds
Epoch  450  loss  10.761804663278433 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  460  loss  9.712251116711883 correct 48
Time for last 10 epochs: 0.04 seconds
Epoch  470  loss  9.395179696072635 correct 49
Time for last 10 epochs: 0.04 seconds
Epoch  480  loss  9.163997120043327 correct 47
Time for last 10 epochs: 0.04 seconds
Epoch  490  loss  9.238361521077316 correct 48
Time for last 10 epochs: 0.05 seconds

GPU XOR Data:
Epoch  0  loss  48.14213291676662 correct 24
Time for last 10 epochs: 0.67 seconds
Epoch  10  loss  32.73963399837454 correct 26
Time for last 10 epochs: 0.04 seconds
Epoch  20  loss  31.47806929988109 correct 33
Time for last 10 epochs: 0.04 seconds
Epoch  30  loss  28.432804458691237 correct 38
Time for last 10 epochs: 0.04 seconds
Epoch  40  loss  27.872725657063576 correct 36
Time for last 10 epochs: 0.04 seconds
Epoch  50  loss  27.40443598782391 correct 39
Time for last 10 epochs: 0.07 seconds
Epoch  60  loss  26.884364651197473 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  70  loss  26.185563994464403 correct 42
Time for last 10 epochs: 0.05 seconds
Epoch  80  loss  25.02383960229396 correct 41
Time for last 10 epochs: 0.05 seconds
Epoch  90  loss  24.780549598144578 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  100  loss  23.857378011392747 correct 44
Time for last 10 epochs: 0.07 seconds
Epoch  110  loss  23.874221455411906 correct 39
Time for last 10 epochs: 0.04 seconds
Epoch  120  loss  23.36848241721976 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  130  loss  23.085575349434393 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  140  loss  22.21240723050054 correct 42
Time for last 10 epochs: 0.04 seconds
Epoch  150  loss  21.222350428335538 correct 42
Time for last 10 epochs: 0.07 seconds
Epoch  160  loss  20.9995017329215 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  170  loss  20.801099833360368 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  180  loss  20.834994755153478 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  190  loss  19.948830219833027 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  200  loss  19.29809503761005 correct 41
Time for last 10 epochs: 0.08 seconds
Epoch  210  loss  19.1121604761553 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  220  loss  18.586406219566797 correct 42
Time for last 10 epochs: 0.05 seconds
Epoch  230  loss  18.98696103661834 correct 42
Time for last 10 epochs: 0.04 seconds
Epoch  240  loss  17.61675263192149 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  250  loss  18.67044149369776 correct 43
Time for last 10 epochs: 0.07 seconds
Epoch  260  loss  17.51810320505591 correct 41
Time for last 10 epochs: 0.04 seconds
Epoch  270  loss  17.080429844009117 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  280  loss  16.81568686506478 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  290  loss  15.82170201250379 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  300  loss  17.073173492951767 correct 45
Time for last 10 epochs: 0.07 seconds
Epoch  310  loss  14.283111616499067 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  320  loss  14.9566448185554 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  330  loss  14.6814505800946 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  340  loss  14.784832774026416 correct 43
Time for last 10 epochs: 0.04 seconds
Epoch  350  loss  14.418546226438707 correct 44
Time for last 10 epochs: 0.07 seconds
Epoch  360  loss  14.206113195810005 correct 44
Time for last 10 epochs: 0.04 seconds
Epoch  370  loss  13.99311417728359 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  380  loss  13.980797414879072 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  390  loss  13.48290396572319 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  400  loss  13.73478071126888 correct 47
Time for last 10 epochs: 0.07 seconds
Epoch  410  loss  13.044519451781909 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  420  loss  12.193493507521383 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  430  loss  12.364855845322635 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  440  loss  11.88088834726814 correct 45
Time for last 10 epochs: 0.04 seconds
Epoch  450  loss  12.447650972013864 correct 46
Time for last 10 epochs: 0.06 seconds
Epoch  460  loss  11.472237472868628 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  470  loss  11.789214797575173 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  480  loss  11.345056415077671 correct 46
Time for last 10 epochs: 0.04 seconds
Epoch  490  loss  11.634528253323364 correct 46
Time for last 10 epochs: 0.04 seconds


GPU XOR Data (larger size with 180):
Epoch  0  loss  41.27196494692775 correct 33
Time for last 10 epochs: 0.51 seconds
Epoch  10  loss  28.351172475373374 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  20  loss  20.377346406554647 correct 41
Time for last 10 epochs: 0.05 seconds
Epoch  30  loss  18.597788859407004 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  40  loss  17.245027636493496 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  50  loss  16.427215500592347 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  60  loss  18.73525235476347 correct 45
Time for last 10 epochs: 0.07 seconds
Epoch  70  loss  15.929833480230329 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  80  loss  18.120172671367236 correct 43
Time for last 10 epochs: 0.05 seconds
Epoch  90  loss  13.779214226011046 correct 44
Time for last 10 epochs: 0.06 seconds
Epoch  100  loss  14.72232063629511 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  110  loss  13.722098128287744 correct 45
Time for last 10 epochs: 0.07 seconds
Epoch  120  loss  14.864580927003383 correct 44
Time for last 10 epochs: 0.06 seconds
Epoch  130  loss  14.084162941579343 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  140  loss  14.44198396984848 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  150  loss  11.045071734971529 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  160  loss  12.44526732910609 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  170  loss  12.754873671620524 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  180  loss  12.840512244106966 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  190  loss  12.3904558765574 correct 44
Time for last 10 epochs: 0.05 seconds
Epoch  200  loss  10.121173058879727 correct 47
Time for last 10 epochs: 0.07 seconds
Epoch  210  loss  10.486883612813433 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  220  loss  11.235566349238812 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  230  loss  10.124243343479975 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  240  loss  9.559786936594678 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  250  loss  9.70525819124593 correct 46
Time for last 10 epochs: 0.07 seconds
Epoch  260  loss  10.722112010250736 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  270  loss  9.317486223572189 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  280  loss  9.914362831627551 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  290  loss  8.952253953922604 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  300  loss  8.589406378727437 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  310  loss  8.733529107064333 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  320  loss  9.006689447846147 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  330  loss  8.590174221327613 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  340  loss  8.382565502895066 correct 46
Time for last 10 epochs: 0.07 seconds
Epoch  350  loss  9.692386684103692 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  360  loss  8.253179797172596 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  370  loss  10.566244459735612 correct 46
Time for last 10 epochs: 0.06 seconds
Epoch  380  loss  8.712909892238269 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  390  loss  6.9076892761528335 correct 47
Time for last 10 epochs: 0.05 seconds
Epoch  400  loss  7.390610536143698 correct 45
Time for last 10 epochs: 0.05 seconds
Epoch  410  loss  7.29807728618124 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  420  loss  7.063667679237637 correct 46
Time for last 10 epochs: 0.05 seconds
Epoch  430  loss  6.7443058112986485 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  440  loss  6.954992501875875 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  450  loss  6.945928675456788 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  460  loss  6.505361966749106 correct 48
Time for last 10 epochs: 0.05 seconds
Epoch  470  loss  6.770492693955397 correct 49
Time for last 10 epochs: 0.05 seconds
Epoch  480  loss  7.077915404700228 correct 49
Time for last 10 epochs: 0.07 seconds
Epoch  490  loss  6.5422961476697 correct 47
Time for last 10 epochs: 0.05 seconds