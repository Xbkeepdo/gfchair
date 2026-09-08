# Audit failure and recovery history

- Image `571678`: FP32 audit failed while duplicating the decoder layer on a 24 GiB GPU. Recovery changed the detailed-case FP32 audit to temporarily convert only the current layer's norm and FFN in place, then restore the original dtype. Completed image shards were retained.
- Image `213525`: FP32 Gauss–Legendre K64 failed at token chunk 256 with CUDA OOM. Recovery applied the preregistered `256/128/64/32` fallback to the complete audit path. Completed image shards were retained.
- Both incidents were infrastructure/memory failures, not silently skipped cases. The formal audit resumed from the first missing image and must finish with 200 processed images, 50 detailed cases, and zero remaining failures before acceptance.
