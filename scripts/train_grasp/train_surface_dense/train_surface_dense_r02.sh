PY_NAME="${HOME_DIR}/experiments/train.py"
SEED="42"

# Main cnfigurations
NET_CONFIG="${HOME_DIR}/config/net_config/grasp_dus2.yaml"
OPTIM_CONFIG="${HOME_DIR}/config/optim_config/optim_cd_sparse.yaml"
TRAIN_DATA_CONFIG="${HOME_DIR}/config/data_config/modelnet_voxel_dense.yaml train_cfg"
VAL_DATA_CONFIG="${HOME_DIR}/config/data_config/modelnet_voxel_dense.yaml val_cfg"

# Method-specific parameters
ALPHA="5" # distortion trade-off
BETA="2" # rate trade-off
SCALING_RATIO="0.25" # quantization ratio
POINT_MUL="10" # point multiplication, also the number of neighbors to search
SKIP_MODE="False" # skip mode
SEED="42"

# Logging settings
PRINT_FREQ="20"
PC_WRITE_FREQ="-1"
TF_SUMMARY="True"
SAVE_CHECKPOINT_FREQ="1"
SAVE_CHECKPOINT_MAX="10"
VAL_FREQ="-1"
VAL_PRINT_FREQ="20"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
