import torch
class TensorReshaper:
    def __init__(self, N):
        self.n = None
        self.N = N
        self.__x_transformed = None

    def reshape_tensor(self, x):
        self.b, c, self.n, h, w = x.shape

        if self.n % self.N != 0:
            raise ValueError(f"n ({self.n}) must be divisible by N ({self.N})")

        self.__x_transformed = x.transpose(1, 2).contiguous().view(-1, self.N, c, h, w).transpose(1, 2)

        return self.__x_transformed

    def get_transformed(self):
        return self.__x_transformed

    def restore_all_batches(self):
        x_restored = torch.cat([self.restore_batch(i) for i in range(self.b)], dim=0)

        return x_restored

    def restore_batch(self, batch_index):
        if batch_index < 0 or batch_index >= self.b:
            raise ValueError("Batch index out of range")

        if self.__x_transformed is None:
            raise ValueError("reshape_tensor method must be called before restore_batch")

        start_index = batch_index * self.n // self.N
        end_index = (batch_index + 1) * self.n // self.N

        batch = self.__x_transformed[start_index:end_index]
        _, c, _, h, w = batch.shape
        batch = batch.contiguous().view(1, -1, c, self.N, h, w)
        batch = batch.transpose(1, 2)
        batch = batch.contiguous().view(1, c, self.n, h, w)

        return batch



if __name__=="__main__":
    import numpy as np
    from tqdm import tqdm

    # 检查是否有可用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 指定测试用例的数量
    num_tests = 100

    # 创建一个进度条
    pbar = tqdm(total=40 * num_tests)

    # 对于每一个 N 的值
    for N in range(1, 41):
        # 创建一个 TensorReshaper 实例
        reshaper = TensorReshaper(N)

        # 对于每一个测试用例
        for i in range(num_tests):
            # 随机生成张量的形状，保证五维，但是具体形状也是随机的
            shape = np.random.randint(low=N, high=N+10, size=5).tolist()

            # 生成一个随机张量，并将其移动到 GPU 上
            x = torch.randn(shape, device=device)

            # 确保张量的第三维度可以被 N 整除
            if x.shape[2] % reshaper.N != 0:
                continue

            # 使用 reshape_tensor 方法来改变张量的形状
            x_transformed = reshaper.reshape_tensor(x)

            # 使用 restore_all_batches 方法来恢复原始的张量形状
            x_restored = reshaper.restore_all_batches()

            # 检查恢复后的张量形状是否与原始的张量形状相同
            assert x.shape == x_restored.shape, f"Test {i} with N={N}: The restored tensor does not have the same shape as the original tensor."

            # 检查恢复后的张量与原始的张量是否相同
            assert torch.allclose(x,
                                  x_restored), f"Test {i} with N={N}: The restored tensor is not the same as the original tensor."

            # 更新进度条
            pbar.update(1)

    # 完成进度条
    pbar.close()

    print("All tests passed!")




