PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
CHECKPOINTS="${HOME_DIR}/results/train_grasp_quad_r01/epoch_newest.pth"
CHECKPOINT_NET_CONFIG="True"
#NET_CONFIG="${HOME_DIR}/config/net_config/grasp_quad_res.yaml"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/grasp_surface.yaml"
INPUT="${HOME_DIR}/datasets/cat1/A/soldier_viewdep_vox12.ply ${HOME_DIR}/datasets/cat1/A/boxer_viewdep_vox12.ply ${HOME_DIR}/datasets/cat1/A/Facade_00009_vox12.ply ${HOME_DIR}/datasets/cat1/A/House_without_roof_00057_vox12.ply"
COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv"
WRITE_PREFIX="grasp_quad"
PRINT_FREQ="1"
PC_WRITE_FREQ="1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
PEAK_VALUE="4095 4095 4095 4095"
BIT_DEPTH="12 12 12 12"
SLICE="0"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
