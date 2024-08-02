import torch

class DataAugmentor:
    def __init__(self, random_attack_num_samples, perturb_epsilon):
        self.random_attack_num_samples = random_attack_num_samples
        self.perturb_epsilon = perturb_epsilon

    @torch.no_grad()
    def augment_data_y(self, embed: torch.Tensor, y: torch.Tensor, sens_attr_vectors: dict):
        assert y.dim() == 1 and sens_attr_vectors is not None

        y_repeated = y.repeat_interleave(self.random_attack_num_samples)
        assert embed.dim() == 2 and embed.size(0) == y.size(0)

        noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
        coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon

        sens_attr_vector_tensor = torch.zeros_like(noisy_latents)
        for label in sens_attr_vectors.keys():
            mask = (y_repeated == label)
            sens_attr_vector_tensor[mask] = sens_attr_vectors[label]
        print(sens_attr_vector_tensor)

        noisy_latents += sens_attr_vector_tensor * coeffs

        return noisy_latents, y_repeated

# 设置参数
random_attack_num_samples = 5
perturb_epsilon = 0.1

# 初始化DataAugmentor类
augmentor = DataAugmentor(random_attack_num_samples, perturb_epsilon)

# 生成随机数据来模拟嵌入表示、标签和敏感属性向量
torch.manual_seed(42)  # For reproducibility
num_samples = 10
embedding_dim = 8

embed = torch.randn(num_samples, embedding_dim)
y = torch.randint(0, 2, (num_samples,))  # 生成0, 1, 2之间的随机标签

# 模拟敏感属性向量字典
sens_attr_vectors = {
    0: torch.randn(embedding_dim),
    1: torch.randn(embedding_dim),
   
}
print(sens_attr_vectors)

# 调用数据增强方法
noisy_latents, y_repeated = augmentor.augment_data_y(embed, y, sens_attr_vectors)

# 打印结果
print("Original Embeddings:\n", embed)
print("Labels:\n", y)
print("Noisy Latents:\n", noisy_latents)
print("Repeated Labels:\n", y_repeated)
