import torch
import torch_npu
import triton
import triton.language as tl
import time


# 极速版：直接去掉 Mask，假设 N 刚好是 2 的幂（这里是 1024）
@triton.jit
def softmax_fast_kernel(output_ptr, input_ptr, n_rows, n_cols, row_stride, BLOCK_SIZE: tl.constexpr):
    # 每个 Program 处理 1 行
    row_idx = tl.program_id(0)

    # 列索引，不需要 Mask
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 加载当前行的数据
    input_ptrs = input_ptr + row_idx * row_stride + col_offsets
    row = tl.load(input_ptrs)

    # 核心计算：减最大值 -> Exp -> 求和 -> 除
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 写回
    output_ptrs = output_ptr + row_idx * row_stride + col_offsets
    tl.store(output_ptrs, softmax_output)


def triton_softmax_fast(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # 一个 Program 处理一行，Grid 大小就是行数
    grid = (n_rows,)

    # BLOCK_SIZE 直接固定为 1024（与列数对齐）
    softmax_fast_kernel[grid](output, x, n_rows, n_cols, x.stride(0), BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 1024
    x = torch.randn(M, N, device='npu', dtype=torch.float32)

    print("--- 精度测试 ---")
    triton_out = triton_softmax_fast(x)
    torch_out = torch.softmax(x, dim=-1)
    if torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4):
        print("✅ 极速版测试通过！")

    print("\n--- 性能测试 ---")


    def benchmark(func, *args, **kwargs):
        torch.npu.synchronize()
        start = time.time()
        for _ in range(50):
            func(*args, **kwargs)
        torch.npu.synchronize()
        return (time.time() - start) / 50


    # 预热
    triton_softmax_fast(x)
    torch.softmax(x, dim=-1)

    t_triton = benchmark(triton_softmax_fast, x)
    t_torch = benchmark(torch.softmax, x, dim=-1)

    print(f"Triton 极速版耗时: {t_triton * 1000:.3f} ms")
    print(f"PyTorch 耗时: {t_torch * 1000:.3f} ms")
    print(f"性能对比: {t_torch / t_triton:.2f}x")