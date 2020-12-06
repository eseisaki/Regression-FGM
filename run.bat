@ECHO OFF
SETLOCAL
SET CHOICE=3
SET NEW_DATASET=drift
SET K=1
SET POINTS=140000
SET EPOCH=4
SET VAR=10
SET FEATURES=10
SET VPER=0.25
SET ERROR=0.07
SET WIN_SIZE=14000
SET WIN_STEP=1
SET TEST=False
SET DEBUG=False
SET IN_FILE=io_files/drift
SET OUT_FILE=io_files/fgm_nodes_1_features_10_error_0.05_epoch_4
             io_files/gm/drift/nodes/regret/k0_ft10_e5_win10000_ep4.csv'

python main.py %CHOICE% %NEW_DATASET% %K% %POINTS% %EPOCH% %VAR% %FEATURES% %VPER% %ERROR% %WIN_SIZE% %WIN_STEP% %TEST% %DEBUG% %IN_FILE% %OUT_FILE%

PAUSE