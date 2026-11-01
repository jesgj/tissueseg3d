# Third-party notices

This project is distributed under the BSD 3-Clause licence (see `LICENSE`).
Parts of the numerical core are ports of source code from other BSD-licensed
projects, listed below.

## NumPy

`src/numpy_compat.rs` reproduces the arithmetic of two NumPy components so that
the port produces bit-identical results on quantised output:

* `pairwise_sum` ports `DOUBLE_pairwise_sum` from
  `numpy/_core/src/umath/loops_utils.h.src`, the summation order used by
  `numpy.sum` on contiguous `float64` data.
* `exp` ports the AVX512 normal path of the bundled SVML `__svml_exp8_ha`
  routine from `numpy/_core/src/umath/svml` (SVML revision
  `bd9d46e0c627ab16f3d2641f7cdb1bc15d303ba4`, file
  `linux/avx512/svml_z0_exp_d_ha.s`), which is what `numpy.exp` dispatches to
  for `float64` arrays on AVX512 hosts. The polynomial coefficients and the
  `2^(j/16)` / residual table entries are preserved bit for bit.

The kernel layout, magic constants and evaluation order are taken from that
assembly; the resulting Rust code is a portable scalar translation, not a copy
of the assembly.

Upstream: <https://github.com/numpy/numpy> and <https://github.com/numpy/SVML>

```
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The SVML source distributed with NumPy carries the same BSD 3-Clause terms
with the header `Copyright (c) 2005-2021, NumPy Developers. All rights
reserved.` and Intel's notice that the software is provided under the licence
referenced in `https://github.com/numpy/SVML`.

The adapted SVML assembly also retains this upstream copyright notice:

```
Copyright 1996-2023 Intel Corporation.
```
