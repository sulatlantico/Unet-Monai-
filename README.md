# 🧠 UNet 2D com MONAI & PyTorch Lightning

**Segmentação semiautomática de dígitos MNIST usando arquitetura UNet 2D com MONAI, PyTorch e PyTorch Lightning**

---

## 📚 Bibliotecas Utilizadas

- [MONAI](https://monai.io/) – módulo `monai.networks.nets.UNet` para construir a UNet 2D :contentReference[oaicite:1]{index=1}  
- PyTorch – backend de deep learning  
- PyTorch Lightning – estrutura para organizar treinamento com menos boilerplate  
- torchvision – para carregar o dataset MNIST :contentReference[oaicite:2]{index=2}  
- Python (numpy, os, datetime…)

---

## 🧩 Conjunto de Dados

- **Dataset**: MNIST  
  - Imagens de dígitos manuscritos (28×28), rotuladas de 0 a 9  
  - Foi utilizado como proxy para simular segmentação: cada dígito vira uma **máscara binária**  
  - Pré-processamento realizado em `process_data.py`

---

## 🧪 Exemplos do Dataset

![MNIST Examples](examples/mnist_examples.png)

Amostras do conjunto de dados MNIST utilizado para simular segmentação binária.


## 🏗️ Arquitetura da Rede

O modelo UNet 2D foi definido com:

- `spatial_dims=2`, `in_channels=1`, `out_channels=1`
- **Camadas**: `(4, 8, 16)` filtros progressivamente em encoder/decoder
- **Strides**: `(2, 2)` — dois down/up-samples de fator 2
- `num_res_units=0` (sem residuais), ativação `PReLU`, normalização `InstanceNorm`
- Usando `BasicUNet` do MONAI

A definição segue este bloco:

```python
net = UNet(
  spatial_dims=2, in_channels=1, out_channels=1,
  channels=(4, 8, 16), strides=(2, 2), num_res_units=0
)
