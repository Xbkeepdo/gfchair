# Audit failure and recovery history

- Image `150779`: the FP32 Gauss–Legendre K64 audit exhausted the preregistered token chunks `256/128/64/32` because captures for all 36 decoder layers were still resident although the audit used only four representative layers.
- Recovery released non-representative layer captures immediately after the forward pass. Completed image shards were retained, the failed image was retried, and the audit resumed without skipping a case.
- Final acceptance is based on the resumed PASS artifacts: 200 processed images, 50 detailed FP32/Shapley cases, zero remaining failures, and complete checksums.
