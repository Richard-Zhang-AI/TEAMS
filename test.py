import sys
sys.argv.extend(['--cfg_file', 'configs/sbd_snake.yaml', 'ct_score', '0.4', 'train_or_test', 'test'])

from run import run_test_medical
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
run_test_medical()