import torch
import torch_npu
import triton
import triton.language as tl
import time


# 使用 Autotune 自动寻找最优的分块大小
@triton.autotune(
    configs=[
        # 尝试不同的分块组合
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # 矩阵指针
        a_ptr, b_ptr, c_ptr,
        # 矩阵维度
        M, N, K,
        # 步长 (Stride)
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # 分块大小 (编译期常量)
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 1. 计算当前 Program 处理的是哪个分块 (pid_m, pid_n)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 2. 计算当前分块的 M 和 N 的偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # K 维度的偏移量
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化 A 和 B 的指针
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 3. 在 K 维度上进行循环累加
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A 和 B 的分块（加上 Mask 防止越界）
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # 核心计算：矩阵乘加
        accumulator += tl.dot(a, b)

        # 指针移动到下一个 K 块
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. 将结果写回全局内存
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def triton_matmul(a, b):
    # 使用 float16 进行计算，这是大模型最常见的精度
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 计算网格大小
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    # 文档要求的 Shape: 1024, 16384, 1024
    M, K, N = 1024, 16384, 1024
    print(f"正在生成张量: A({M}x{K}), B({K}x{N})...")

    # 使用 float16 减少内存占用，加快运算 (Ascend 对 fp16 支持极好)
    a = torch.randn((M, K), device='npu', dtype=torch.float16)
    b = torch.randn((K, N), device='npu', dtype=torch.float16)

    print("--- 精度测试 ---")
    triton_out = triton_matmul(a, b)
    torch_out = torch.matmul(a, b)

    # 矩阵乘误差允许范围稍微放宽 (fp16 的特性)
    if torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2):
        print("✅ Matmul 测试通过！与 PyTorch 结果一致。")
    else:
        print("❌ 测试失败！")
        diff = (triton_out - torch_out).abs().max().item()
        print(f"最大误差: {diff}")

    print("\n--- 性能测试 (因为 K 很大，跑 20 次即可) ---")


    def benchmark(func, *args, **kwargs):
        torch.npu.synchronize()
        start = time.time()
        for _ in range(20):
            func(*args, **kwargs)
        torch.npu.synchronize()
        return (time.time() - start) / 20


    # 预热 (第一次编译和 autotune 会花一点时间，耐心等几秒)
    print("预热并寻找最优分块配置中，请稍候...")
    triton_matmul(a, b)
    torch.matmul(a, b)

    t_triton = benchmark(triton_matmul, a, b)
    t_torch = benchmark(torch.matmul, a, b)

    print(f"Triton 耗时: {t_triton * 1000:.3f} ms")
    print(f"PyTorch 耗时: {t_torch * 1000:.3f} ms")
    print(f"性能对比: {t_torch / t_triton:.2f}x")