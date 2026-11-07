# Checkpoints

This directory is intended to contain model checkpoints when running locally. On Kaggle, the notebook will load checkpoints from `/kaggle/input/...`.

Expected files:
- Primary TemporalUNet: `biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth`
- Secondary TemporalUNet: `biohub-temporal-unet3d-seed314159-v1/secondary_seed_weights/.../edge_predictor_best.pth`
- DeepCenter Prior: `biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt`
