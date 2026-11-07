# Evaluating VLM Architectures for Low-Level Image Quality Diagnostics

Course project for **DA624: NLP with Large Language Models (2026)**.

## About the Project

Vision-Language Models (VLMs) work well on high-level questions like "What is in this picture?", but they often fail on low-level questions like "Is this image blurry or noisy?". Normal quality scores (PSNR, SSIM, LPIPS) give one number but do not explain *why* an image looks bad.

In this project we test how the way a model joins image and text (called **fusion**) affects its ability to judge image quality. We compare three fusion types:

| Fusion type | Model |
|---|---|
| Early fusion | ViLT |
| Mid fusion | BLIP-2 |
| Token-level fusion | LLaVA |

## Contributors

| Name | Roll No. | Contribution |
|---|---|---|
| Yash Kharangate | 230150032 | Early fusion (ViLT) |
| Billa Cherish | 230150007 | Mid Fusion (BLIP-2) |
| Vaishnavi Agarwal | 230150028 | Token Level Fusion (LLaVa) |
