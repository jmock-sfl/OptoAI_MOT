from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import time

config_file = 'configs/solov2_light_448_r34_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/solov2_light_r34_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
start = time.monotonic()
# test a single image
img = 'demo/demo.jpg'
result = inference_detector(model, img)
stop = time.monotonic()

print('EXECUTION TIME:', stop-start)

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")
