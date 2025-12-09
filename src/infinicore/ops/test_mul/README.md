# 示例算子 `test_mul` 开发全流程（向量点积）

本文件通过一个非常简单的算子 **`test_mul`（向量点积）**，完整演示在 InfiniCore 中
从 **C++ 接口 → Kernel 实现 → Dispatcher 注册 → Python 绑定 → Python 测试**
的开发流程，方便你后续实现自己的算子时照着抄一遍。

> 说明：为突出流程，本示例在功能上刻意做了简化：
> - **只支持 CPU + float32**
> - **只支持一维连续向量的点积**
> - 输出为 **0 维标量 Tensor（shape = {}）**

---

## 1. C++ 算子接口定义（头文件）

位置：`include/infinicore/ops/test_mul.hpp`

关键内容：

- 定义算子类 `TestMul`，给出：
  - `using schema = void (*)(Tensor, Tensor, Tensor);`
  - `static void execute(Tensor out, Tensor a, Tensor b);`
  - `static common::OpDispatcher<schema> &dispatcher();`
- 提供对外 C++ 接口：
  - `Tensor test_mul(Tensor a, Tensor b);`
  - `void test_mul_(Tensor out, Tensor a, Tensor b);`

这一步的作用是：**确定算子的调用方式和 kernel 函数签名**，后续所有实现都围绕这个
`schema` 和这些接口展开。

---

## 2. C++ 算子实现与 Dispatcher（通用执行逻辑）

位置：`src/infinicore/ops/test_mul/test_mul.cc`

职责：

- 为 `TestMul` 创建一个 **全局 dispatcher 实例**：
  - 使用静态局部变量 `static common::OpDispatcher<TestMul::schema> dispatcher_;`
- 在 `execute` 中做：
  - **检查设备一致性**：`INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b);`
  - **设置当前运行设备**：`context::setDevice(out->device());`
  - **通过 dispatcher 分发到具体设备 kernel**：
    - `dispatcher().lookup(out->device().getType())(out, a, b);`
- 在对外接口 `test_mul` 中：
  - 假定输入 `a, b` 设备一致、dtype 一致；
  - 构造一个 **0 维标量 Tensor** 作为输出：
    - `Shape scalar_shape{};`
    - `auto out = Tensor::empty(scalar_shape, dtype, device);`
  - 调用 `test_mul_(out, a, b);`

> 要点：
> - `test_mul.cc` **不关心**具体怎么计算点积，它只负责：
>   - 规范好输入输出；
>   - 把调用转交给 dispatcher 中注册的 kernel。
> - 真正的“乘加循环”等逻辑在下一步 kernel 中实现。

---

## 3. 设备相关 Kernel 实现与注册（以 CPU 为例）

位置：`src/infinicore/ops/test_mul/test_mul_cpu.cc`

在这个文件中我们做两件事：

1. **实现具体 kernel 函数 `calculate`**
2. **在静态初始化阶段把 kernel 注册到 dispatcher**

### 3.1 kernel 函数 `calculate`

签名：`void calculate(Tensor out, Tensor a, Tensor b);`，与 `schema` 一致。

示例逻辑：

- 限制条件（示例中通过 `if` + `throw` 明确约束）：
  - 设备：`out->device().getType() == Device::Type::CPU`
  - dtype：`out/a/b` 都是 `DataType::F32`
  - `a`、`b`：
    - `ndim() == 1`
    - `shape()` 相同
    - `is_contiguous() == true`
  - `out->numel() == 1`（标量）
- 真正的点积：
  - `Size n = a->numel();`
  - 通过 `reinterpret_cast<const float *>(a->data())` 拿到裸指针；
  - 一个 `for` 循环累加 `acc += a_ptr[i] * b_ptr[i];`
  - 把结果写入 `out_ptr[0]`。

### 3.2 在 dispatcher 中注册 CPU kernel

同一文件末尾通过静态变量进行注册：

```c++
static bool registered = []() {
    TestMul::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();
```

含义是：

- 当 `infinicore` 动态库被加载时，这个 lambda 会自动执行；
- 把 `calculate` 注册到 `TestMul` 的 dispatcher 中，绑定到设备类型 `CPU`；
- 之后任何调用 `TestMul::execute`，只要输出张量在 CPU 上，就会走到这个 kernel。

> 提示：
> - 如果你有多个后端（如 NVIDIA / Ascend / MLU 等），可以各写一个
>   `<op_name>_<device>.cc`，在各自文件中注册对应设备的 kernel。

---

## 4. Python 绑定与前端接口

这一部分分三层：

1. **pybind11 绑定 C++ 接口**（暴露到 `_infinicore` 模块）
2. **Python 包装函数**（提供友好的 Python API）
3. **在 `infinicore.__init__` 中导出统一接口**

### 4.1 pybind11 绑定

- 位置：`src/infinicore/pybind11/ops/test_mul.hpp`
- 提供函数：
  - `void bind_test_mul(py::module &m);`
- 在函数内部注册两个 C++ 接口：
  - `_infinicore.test_mul(a, b)`
  - `_infinicore.test_mul_(out, a, b)`
- 再到 `src/infinicore/pybind11/ops.hpp` 中：
  - `#include "ops/test_mul.hpp"`
  - 在 `inline void bind(py::module &m)` 里调用 `bind_test_mul(m);`

这样，构建出的 Python 扩展 `_infinicore` 模块就拥有了 `test_mul/test_mul_` 两个
函数。

### 4.2 Python 前端包装

- 位置：`python/infinicore/ops/test_mul.py`
- 通过 `_infinicore` 模块封装出 Python 函数：

```python
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def test_mul(a: Tensor, b: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.test_mul(a._underlying, b._underlying))

    _infinicore.test_mul_(out._underlying, a._underlying, b._underlying)
    return out
```

### 4.3 在 `infinicore` 顶层导出

- 位置：`python/infinicore/__init__.py`
- 两处改动：
  1. `from infinicore.ops.test_mul import test_mul`
  2. 在 `__all__` 中加入 `"test_mul",`

这样在用户侧就可以直接：

```python
import infinicore

out = infinicore.test_mul(a, b)
```

---

## 5. Python 端单元测试

位置：`test/infinicore/ops/test_mul.py`

遵循现有测试框架（如 `add.py`）的写法，主要步骤：

1. 准备测例配置 `_TEST_CASES_DATA`：
   - 这里只测试若干一维 shape，例如 `(8,)`, `(32,)`, `(1024,)`。
2. 限定 dtype：
   - `_TENSOR_DTYPES = [infinicore.float32]`（和 kernel 限制一致）。
3. 实现 `parse_test_cases()`：
   - 使用 `TensorSpec` 构造输入张量规格；
   - 仅构造 out-of-place 调用（返回值形式）。
4. 继承 `BaseOperatorTest`：
   - `get_test_cases()` 返回构造好的测例；
   - `torch_operator()` 使用 `torch.dot` 作为参考实现；
   - `infinicore_operator()` 调用 `infinicore.test_mul`。
5. 使用 `GenericTestRunner` 统一跑测：
   - 命令示例：

```bash
python test/infinicore/ops/test_mul.py --nvidia --verbose --bench
```

> 提示：
> - 由于本示例算子只实现了 CPU kernel，实际测试时只需要确保 CPU 测试通过即可。
> - 如果你后续为 GPU / 其他后端补充 kernel，测试脚本可以保持不动。

---

## 6. 开发自己算子的推荐步骤（可直接复用）

当你要实现一个新算子 `<op_name>` 时，可以套用下面的流程：

1. **C++ 头文件**：`include/infinicore/ops/<op_name>.hpp`
   - 定义 `<OpName>` 类：`schema` / `execute` / `dispatcher`
   - 定义对外接口函数：`<op_name>(...)` 和 `<op_name>_(...)`
2. **通用执行逻辑**：`src/infinicore/ops/<op_name>/<op_name>.cc`
   - 拷贝本示例中 `test_mul.cc` 的结构：
     - 设备检查、`context::setDevice`、`dispatcher().lookup(...)`
     - out-of-place 接口里构造输出 Tensor 尺寸，然后调用 in-place 版本
3. **设备相关 kernel**：
   - 每个后端写一个 `src/infinicore/ops/<op_name>/<op_name>_<device>.cc`
   - 在文件中实现 `calculate(...)` 并通过
     `dispatcher().registerDevice(Device::Type::<DEVICE>, &calculate);`
     完成注册
4. **pybind 绑定**：
   - `src/infinicore/pybind11/ops/<op_name>.hpp` 写 `bind_<op_name>`
   - 在 `src/infinicore/pybind11/ops.hpp` 中 `#include` 并调用
5. **Python 前端包装**：
   - `python/infinicore/ops/<op_name>.py` 里用 `_infinicore.<op_name>` 封装成 Python 函数
   - 在 `python/infinicore/__init__.py` 中导入并加入 `__all__`
6. **Python 测试**：
   - `test/infinicore/ops/<op_name>.py` 里继承 `BaseOperatorTest`
   - 准备若干测例 + PyTorch 参考实现 + InfiniCore 调用

按照上面流程，你可以把 `test_mul` 当作一个最小可运行样例，后续实现复杂算子时直接复制这些文件改名、再逐步丰富逻辑即可。


