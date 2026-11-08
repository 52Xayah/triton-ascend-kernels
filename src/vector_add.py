import torch
import torch_npu
import triton
import triton.language as tl
import time


# 1. 定义 Triton 核函数 (Kernel)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    # 任务1：基础并行与核间切分
    # 获取当前 Program 的 ID
    pid = tl.program_id(axis=0)
    # 计算当前 Program 处理的起始位置
    block_start = pid * XBLOCK

    # 调试技巧：打印当前 Program 的 ID (取消下面这行的注释可以查看)
    # tl.device_print("pid: ", pid)

    # 任务2：引入核内循环切分（XBLOCK_SUB）
    # 将大块 XBLOCK 拆分为多次小块 XBLOCK_SUB 处理
    for i in range(0, XBLOCK, XBLOCK_SUB):
        # 计算当前循环处理的数据偏移量
        offsets = block_start + i + tl.arange(0, XBLOCK_SUB)

        # 任务3：边界安全处理（Mask）
        # 当总元素无法整除时，防止越界访问
        mask = offsets < n_elements

        # 加载数据，对于越界读取，使用 other=0.0 填充
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

        # 执行向量加法
        output = x + y

        # 将结果写回，同样需要掩码保护
        tl.store(output_ptr + offsets, output, mask=mask)


# 2. 定义 Python 调用接口
def add(x, y):
    # 创建一个空的输出 tensor
    output = torch.empty_like(x)
    n_elements = x.numel()

    # 设定 XBLOCK 和 XBLOCK_SUB 的大小
    # 注意：XBLOCK_SUB 必须是 2 的幂，且不能太大导致 UB 溢出
    XBLOCK = 1024 * 4
    XBLOCK_SUB = 1024

    # 计算 Grid 大小（向上取整）
    grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']),)

    # 启动 Kernel
    add_kernel[grid](x, y, output, n_elements, XBLOCK=XBLOCK, XBLOCK_SUB=XBLOCK_SUB)
    return output


# 3. 测试与验证代码
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(0)

    # 创建一个较大的张量，模拟大 shape (故意设置成不能整除的大小，测试 Mask 功能)
    n = 98432
    x = torch.rand(n, device='npu')
    y = torch.rand(n, device='npu')

    # 1. 使用 Triton 计算
    triton_output = add(x, y)

    # 2. 使用 PyTorch 原生算子计算
    torch_output = x + y

    # 3. 验证精度（对比两个结果是否一致）
    if torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-5):
        print("✅ 测试通过！Triton 计算结果与 PyTorch 一致。")
    else:
        print("❌ 测试失败！精度不匹配。")
        print("最大误差: ", (triton_output - torch_output).abs().max().item())