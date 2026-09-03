# Reproducibility Checklist

Một run được gọi là reproducible khi người khác có đủ thông tin để tái tạo pipeline gần nhất có thể, không chỉ có file weight.

## Repo lưu gì?

- config YAML
- pinned UVW dataset revision
- data filter parameters
- tokenizer JSON
- binary data metadata + SHA-256
- model config
- optimizer hyperparameters
- RNG state trong checkpoint
- train/val metrics
- summary test metrics

## Seed

`seed` dùng làm base seed. Repo offset seed riêng cho tokenizer subset, train subset, validation subset và test subset. Validation/test batch sampling có seed riêng trong training config.

## Các nguồn vẫn có thể gây nondeterminism

GPU kernels, fused attention, compiler và library version có thể tạo khác biệt floating-point. “Reproducible” không nên hiểu là mọi GPU mọi PyTorch cho bit-identical weight.

Nếu cần experiment strict hơn:

- dùng manual attention
- pin package versions/container image
- bật deterministic algorithms nếu operation hỗ trợ
- ghi GPU model/CUDA/PyTorch
- không thay dataset revision

## Checkpoint resume

Checkpoint v2 lưu RNG state và tokenizer mapping. Khi publish model, giữ luôn `config.yaml` và tokenizer.

## Artifact manifest

`data_manifest.json` ghi metadata binary train/val/test. Nếu hai run có SHA-256 data stream khác nhau, chúng không phải cùng exact data artifact dù config nhìn giống nhau.
