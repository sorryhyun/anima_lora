# Dynamic Shapes[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#dynamic-shapes "Link to this heading")

Created On: May 19, 2023 | Last Updated On: Jan 09, 2026

This section explains how to work with dynamic shapes in PyTorch, including how
to debug and fix common errors, implement support for dynamic shapes in
operators, and understand the underlying mechanisms.

Dynamic shapes allow PyTorch models to handle inputs with varying dimensions
without recompilation. This enables more flexible models that can process
different batch sizes, sequence lengths, or image dimensions in a single
compiled artifact. Dynamic shapes work by symbolically tracing tensor
dimensions rather than using concrete values, creating a computation
graph that adapts to different input shapes at runtime. By default,
PyTorch assumes all input shapes to be static.

Typically, deep learning compilers only support static shapes, requiring
recompilation for input shape changes. While this approach covers many use cases,
there are situations where this is insufficient:

- **Variable Dimensions** - Batch sizes or sequence lengths vary, such as in
  adaptive batching.
- **Data-Dependent Outputs** - Models produce outputs based on input data,
  like variable bounding boxes in detection models.
- **Sparse Representations** - Processing depends on data-varying sparse structures,
  such as in sparse tensors, jagged tensors, and graph neural networks.

Dynamic shapes do not support dynamic rank programs, programs which input tensors
change in dimensionality, as this is uncommon and unnecessarily complex.

## What does it mean for a size/integer to be dynamic?[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#what-does-it-mean-for-a-size-integer-to-be-dynamic "Link to this heading")

Dynamic shapes allow avoiding recompilations by making certain dimensions or integers
dynamic. For example, if a function `f(x)` is compiled with a static size, it will need
recompilation for different sizes:

Note

For simplicity, this example uses `@torch.compile(dynamic=True)`. Note, that
this option is not recommended due to it being error prone.
For a recommended way of enabling dynamic shapes, see [Enabling Dynamic Behavior](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#enable-dynamic-behavior).

```
import torch
@torch.compile(dynamic=False)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

```
TRACED GRAPH
 ===== __compiled_fn_1_3f7b5565_3ab5_45de_99db_a5cc01ac6e76 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/281359623.py:4 in f, code: return x* x.size()[0]
        mul: "f32[10][1]cpu" = l_x_ * 10;  l_x_ = None
        return (mul,)
TRACED GRAPH
 ===== __compiled_fn_3_a0c9e279_bcb9_402a_b0cd_5f366829787a =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[20][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/281359623.py:4 in f, code: return x* x.size()[0]
        mul: "f32[20][1]cpu" = l_x_ * 20;  l_x_ = None
        return (mul,)
TRACED GRAPH
 ===== __compiled_fn_5_f698d793_469a_4f4e_9f4e_6906bb3d7827 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[30][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/281359623.py:4 in f, code: return x* x.size()[0]
        mul: "f32[30][1]cpu" = l_x_ * 30;  l_x_ = None
        return (mul,)
TRACED GRAPH
 ===== __compiled_fn_7_73755114_739b_44cf_944a_28a287708a60 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[40][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/281359623.py:4 in f, code: return x* x.size()[0]
        mul: "f32[40][1]cpu" = l_x_ * 40;  l_x_ = None
        return (mul,)
```

```
tensor([21.8454, 16.9744, 19.9298, 13.4449, 38.6766, 39.5037,  8.9695,  5.9749,
        27.4326,  6.0912,  1.7484,  7.5521, 15.7071, 36.3244, 26.2428,  5.7731,
        19.6608,  8.7200,  4.5758, 21.1938, 39.0719, 13.8607, 38.3263, 20.7360,
        38.8485, 23.3334, 38.5470, 31.2488, 16.3918, 32.9518, 37.2355,  0.5628,
        22.1515, 23.8200, 39.7935,  0.7339, 38.3826, 16.1436, 20.6168, 20.9372])
```

In the produced output, you can see that four graphs were generated.
See the corresponding [tlparse output](https://docs.pytorch.org/docs/stable/_static/img/dynamic_shapes/tlparse1_dynamic_shapes_false.png)

By making the size dynamic, the function can handle various sizes without recompilation:

```
import torch
@torch.compile(dynamic=True)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

```
TRACED GRAPH
 ===== __compiled_fn_9_2630054d_b3fe_452b_9b57_c82066d0c6d8 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/1046103881.py:4 in f, code: return x* x.size()[0]
        mul: "f32[s77][1]cpu" = l_x_ * s77;  l_x_ = s77 = None
        return (mul,)
```

```
tensor([38.2346, 10.7907, 18.0950, 38.9933, 17.6504,  2.8249, 33.7380,  1.1236,
         2.7603, 21.5060,  1.6823, 20.3451,  8.5638, 36.0698,  9.4643, 24.0418,
        39.6273, 32.4307, 24.6135, 34.1372, 29.7087, 12.7722,  8.7354, 14.7478,
        22.9152, 38.3146, 14.4804, 36.0626, 14.3778,  4.6169, 18.3100,  5.4258,
        13.7955, 37.2143, 31.3580,  4.1336, 30.7544, 19.3394, 38.2092, 28.5652])
```

With dynamic shapes enabled, only one graph is created. See the
corresponding [tlparse output](https://docs.pytorch.org/docs/stable/_static/img/dynamic_shapes/tlparse2_dynamic_shapes_true.png).

While compilation time differences
are minimal for this small example, more complex use cases would show significant
performance improvements.

## What is a specialization?[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#what-is-a-specialization "Link to this heading")

**Specialization** refers to optimizing a computational graph for specific input shapes
by examining shape conditions during control flow. If a branch is taken based on a
shape condition, the graph is tailored for that condition. If a new input doesn’t meet
this condition, the system will recompile the graph.

Specialization allows you to create optimized computational graphs for specific input
shapes, which can significantly improve execution speed.

```
import torch
@torch.compile(dynamic=True)
def f(x):
    if x.size()[0] == 10:
        return x * 10

    if x.size()[0] <= 30:
        return x*200

    return x*x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
f(torch.rand(50))
```

```
TRACED GRAPH
 ===== __compiled_fn_11_ad05fac5_bfb5_4b8a_bb10_c44e576b1d4e =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/953537014.py:5 in f, code: return x * 10
        mul: "f32[10][1]cpu" = l_x_ * 10;  l_x_ = None
        return (mul,)
TRACED GRAPH
 ===== __compiled_fn_13_86d6e4d9_4d1f_4ac3_89d4_acfc122a6117 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/953537014.py:8 in f, code: return x*200
        mul: "f32[s77][1]cpu" = l_x_ * 200;  l_x_ = None
        return (mul,)
TRACED GRAPH
 ===== __compiled_fn_15_a073a78a_3ef3_4174_9385_1b4e5c58f761 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/953537014.py:10 in f, code: return x*x.size()[0]
        mul: "f32[s77][1]cpu" = l_x_ * s77;  l_x_ = s77 = None
        return (mul,)
```

```
tensor([46.1044, 20.2141, 49.1175, 21.5797,  9.3260,  4.3262, 49.9801, 33.4107,
        44.9485, 17.2226, 32.3609, 39.0174, 36.9090, 25.2275, 11.8367,  5.1183,
        19.4817, 32.8483, 15.3607, 26.4785, 12.2597,  2.6305, 43.8364, 37.4335,
         9.0552, 29.1002, 19.9388, 47.8553, 41.2539, 16.4762, 31.4796, 36.3992,
        34.1561, 17.8050,  4.2273, 26.3260, 11.5127, 40.9477,  5.1918, 42.9819,
         9.2644, 15.8166, 14.3917,  2.5316, 26.1734, 28.4019, 21.9617, 47.8863,
        41.2329, 40.4821])
```

In the code above, we specialize that the graph requires an input size of 10, in which
case it will return `x * 10`. If the input size is less than 30, it will return `x * 200`.
In the output, you can see that this creates three graphs.

See the corresponding [tlparse output](https://docs.pytorch.org/docs/stable/_static/img/dynamic_shapes/tlparse3_specialization.png)

This is how graphs created for the above function:

![../../_images/dynamic_shapes_example_specialization.png](./Dynamic Shapes — PyTorch 2.11 documentation_files/dynamic_shapes_example_specialization.png)

## Enabling Dynamic Behavior[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#enabling-dynamic-behavior "Link to this heading")

There are the following ways to make things dynamic:

- [Automatic dynamic](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#automatic-dynamic)
- [User Annotations](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#user-annotations) (preferred)
- [torch.compile (dynamic=true) (Not recommended)](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#torch-compile-dynamic-true) (for testing only)
- [Advanced Options to Control Dynamic Behavior](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html#dynamic-shapes-advanced-control-options) (for advanced use cases)

Read below about each of this options.

### Automatic dynamic[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#automatic-dynamic "Link to this heading")

**Automatic dynamic** is the default behavior where [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile "torch.compile") performs
the initial compilation assuming static shapes are used, while tracking the
input sizes from that first compilation. When a recompile is triggered, it
uses this information to identify which dimensions have changed and marks
those as dynamic for the second compilation.

### User Annotations[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#user-annotations "Link to this heading")

Several APIs allow users to explicitly mark specific inputs
by name or code as dynamic. This is useful for avoiding initial compilations that
would eventually become dynamic with the previous tools. It is also used to mark
elements that do not automatically get marked as dynamic, such as neural network
module parameters, and so on. User annotations are the preferred way to enable
dynamic shapes.

#### `mark_dynamic(tensor, dim, min=min, max=max)`[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#mark-dynamic-tensor-dim-min-min-max-max "Link to this heading")

> ⚠️ **Warning**
>
> `torch._dynamo.mark_dynamic()` must not be called inside any function
> that is being compiled by `torch.compile()` (for example, a model’s
> `forward()` method or any function it calls).
>
> This function is a *tracing-time API*. If it is invoked from within
> compiled code, Dynamo will raise an error such as:
>
> ```
> AssertionError: Attempt to trace forbidden callable
> ```
>
> Copy to clipboard
>
> **Correct usage** is to call `mark_dynamic` on input tensors *before*
> invoking `torch.compile`, for example:
>
> ```
> torch._dynamo.mark_dynamic(x, 0)
> compiled_model = torch.compile(model)
> ```
>
> Copy to clipboard

The `torch._dynamo.mark_dynamic()` function marks a tensor dimension as dynamic and will fail if it
gets specialized. It does not work for integers. Use this function only if you know
all graphs in the frame using this input converge to a single dynamic graph.
Otherwise, you may encounter a misleading constraint violation error.
In such cases, consider using `torch._dynamo.maybe_mark_dynamic()`. Currently,
`torch._dynamo.mark_dynamic()`
does not have precedence over `force_parameter_static_shapes = True` or `force_nn_module_property_static_shapes = True`.

If you know in advance that a particular dimension will be dynamic, you
can avoid the initial recompilation by using `torch._dynamo.mark_dynamic(tensor, dim)()`.
Additionally, if you already know the minimum and maximum possible
values for this dimension, you can specify them with
`torch._dynamo.mark_dynamic(tensor, dim, min=min, max=max)()`.

Here is a quick example:

```
import torch

@torch.compile
def f(x):
    return x * x.size()[0]

x = torch.randn(10)
torch._dynamo.mark_dynamic(x, 0)

# first invocation we give it is a tensor marked as dynamic
f(x)
# rest of these invocations will use dynamically compiled code
f(torch.randn(20))
f(torch.randn(30))
f(torch.randn(40))
```

```
TRACED GRAPH
 ===== __compiled_fn_17_b64f5486_d2a8_436e_820f_db2a46295a09 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/2254004717.py:5 in f, code: return x * x.size()[0]
        mul: "f32[s77][1]cpu" = l_x_ * s77;  l_x_ = s77 = None
        return (mul,)
```

```
tensor([  29.8175,  111.1459,  -25.7738,    9.9570,    2.5569,    0.1816,
         -59.7626,  -13.6417,   11.9608,   27.3592,    2.1712,   22.4780,
         -30.7502,   -9.4850,   26.9717,   19.2169,   38.5006,  -57.6737,
          34.3796,   39.4891,   -2.1283,   29.2981,  -40.8374,   -3.7874,
          29.0608,    9.5240,   -8.8026,   48.3119,   58.0510,   51.4383,
          17.6919,  -12.7288,   -5.6804,  -26.4697,   12.2641,   44.3255,
        -104.3630,   32.9295,    4.2355,   28.8946])
```

#### `maybe_mark_dynamic(tensor, dim)`[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#maybe-mark-dynamic-tensor-dim "Link to this heading")

The `torch._dynamo.maybe_mark_dynamic()` function shares all properties
with `torch._dynamo.mark_dynamic()`
but does not fail if the size gets specialized. Use it for inputs shared by
multiple graphs or if the number of graphs does not converge to one for a specific
frame. For instance, in the example above, use `torch._dynamo.maybe_mark_dynamic()` because graphs
with sizes 0 and 1 will specialize. However, you can use `torch._dynamo.mark_dynamic()` to ensure
you never specialize.

#### `mark_unbacked(tensor, dim)`[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#mark-unbacked-tensor-dim "Link to this heading")

The `torch._dynamo.decorators.mark_unbacked()` function marks a tensor dimension as unbacked. It is unlikely
to be the tool you need, but it could be useful if the specialization occurs inside
a condition `guard_size_oblivious(x)`, and if using it removes the specialization.
Ensure it fixes the specialization and does not introduce a data-dependent error
that converts to a graph break at or before the specialization location
you are trying to avoid. It might be better to use the next option.

#### Dynamic Allow List (`DYNAMIC_SOURCES`)[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#dynamic-allow-list-dynamic-sources "Link to this heading")

Use the evnironmental variable `TORCH_COMPILE_DYNAMIC_SOURCES` to pass a configuration
list of source names to be marked as dynamic. For example:
`TORCH_COMPILE_DYNAMIC_SOURCES=L[‘x’],L[‘y’]`
It’s easiest to find these dynamic source names using the PGO artifact in `tlparse`.
You can copy and paste the dynamic source names from the PGO artifact. This method works
for integers and tensor sizes and has the highest precedence over all other flags
that force static shapes. It will not throw an error if what is marked dynamic
gets specialized or if the provided input does not exist.

Here is an example:

```
import torch

@torch.compile()
def f(x):
     return x * x.size()[0]

with torch.compiler.config.patch(dynamic_sources="L['x']"):
    f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

```
TRACED GRAPH
 ===== __compiled_fn_19_e994e42f_01f4_45f1_b74e_b5e9601108e3 =====
 /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77][1]cpu"):
        l_x_ = L_x_

        # File: /tmp/ipykernel_525/2867773694.py:5 in f, code: return x * x.size()[0]
        mul: "f32[s77][1]cpu" = l_x_ * s77;  l_x_ = s77 = None
        return (mul,)
```

```
tensor([23.8230, 28.2287, 20.4184, 29.9127, 21.7371, 27.3943, 15.2883, 19.6093,
        38.5449, 15.3658,  1.3110, 29.8560, 10.9331, 28.5505, 17.1665,  2.0957,
         1.6608, 20.7692,  5.6446, 13.1564,  8.2021, 20.2733,  1.8070, 17.1188,
        33.7235,  0.8200,  5.0851, 24.0632, 10.2724, 29.9638, 18.8256, 31.5674,
        10.5684, 19.5051,  3.6410, 37.9080, 25.6940, 11.0745, 16.6204, 18.0237])
```

#### `torch.compiler.set_stance ("eager_then_compile")`[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#torch-compiler-set-stance-eager-then-compile "Link to this heading")

At times, identifying the appropriate inputs to mark as dynamic can
be challenging. If you are willing to accept a performance cost for
the first batch, another convenient option is to use the
`eager_then_compile` stances, which automatically determine dynamic
inputs for you. For more information, see [`torch.compiler.set_stance()`](https://docs.pytorch.org/docs/stable/generated/torch.compiler.set_stance.html#torch.compiler.set_stance "torch.compiler.set_stance") and [Dynamic Compilation Control with torch.compiler.set\_stance](https://docs.pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html).

### `torch.compile (dynamic=true)` (Not recommended)[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#torch-compile-dynamic-true-not-recommended "Link to this heading")

This setting forces all sizes and integers to be dynamic, increasing the
chance of encountering dynamic shape bugs. Setting this option is not
recommended due to it being error prone.
It would make every input size dynamic which may result it performance
regressions and ultimately increase compilation time.

PyTorch also provides advanced control options for dynamic shapes, see:
[Advanced Options to Control Dynamic Behavior](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html#dynamic-shapes-advanced-control-options).

## Where Do I Go From Here?[#](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#where-do-i-go-from-here "Link to this heading")

If you encounter a framework code bug or an issue with specialization,
file an issue so it can be reviewed and potentially improved. If the issue
is within your user code, consider whether you are willing to rewrite your
code to avoid it. Determine if it affects correctness or if it’s a redundant
check. If the issue involves a Triton custom kernel with a `constexpr`
argument, evaluate whether you can rewrite it to address the problem.

- [Dynamic Shapes Core Concepts](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html)
- [Troubleshooting Dynamic Shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting.html)
- [Advanced Options to Control Dynamic Behavior](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html)
- [Beyond the Basics](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_beyond_the_basics.html)

See also

- [tlparse documentation](https://github.com/pytorch/tlparse)
- [The dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)