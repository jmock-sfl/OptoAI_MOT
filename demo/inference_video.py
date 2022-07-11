from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import cv2


config_file = 'configs/solov2/solov2_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/solov2_r50_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

videoFileIn = '/home/jmock/'
videoFileOut = '/videos/'

vidcap = cv2.VideoCapture(videoFileIn)
width  = 1920 #int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # get width
height = 1080 #int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # get height
fps  = vidcap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(videoFileOut, fourcc, fps, output_size, True) # save video
success, frame = vidcap.read(0)

result = inference_detector(model, frame)

show_result_ins(img, result, model.CLASSES, score_thr=0.35, out_file="demo_out.jpg")
