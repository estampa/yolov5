import torch.backends.cudnn as cudnn

import queue
import threading

import pafy
import pyfakewebcam

from models.experimental import *
from utils.datasets import *
from utils.utils import *

frames_raw = queue.Queue()
frames_spaced = queue.Queue()


class LiveStreamThread(Thread):
    def __init__(self, imgsz):
        Thread.__init__(self)
        self.imgsz = imgsz

        url = 'https://youtu.be/51djMAqsmIQ'  # SH63YaIWyK0
        url_pafy = pafy.new(url)
        # print(url_pafy)
        print(url_pafy.streams)
        videoplay = url_pafy.getbest(preftype="mp4")  # webm

        self.cap = cv2.VideoCapture(videoplay.url)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(self.fps, self.size)

    def run(self):
        while True:
            ret, frame0 = self.cap.read()
            if not ret:
                break

            frame = letterbox(frame0, new_shape=self.imgsz)[0]

            frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            frame = np.ascontiguousarray(frame)
            frames_raw.put((frame, frame0))

            # TODO: Empty sometimes after receiving a burst of frames?

        self.cap.release()


class FrameSpacingThread(Thread):
    def __init__(self, fps):
        Thread.__init__(self)
        self.fps = fps

    def run(self):
        # TODO: avoid adding small delays to time
        while True:
            start_time = time.time()

            frame = frames_raw.get(block=True)
            if not frames_spaced.empty():
                frames_spaced.get_nowait()
            frames_spaced.put(frame)

            wait_time = max(1/self.fps - (time.time() - start_time), 0.000000001)
            time.sleep(wait_time)
            # print("my thread %.03f %.03f %0.3f" % (time.time(), wait_time, time.time() - last_time))
            # print("my thread %d" % frames_raw.qsize())


def detect_livestream(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, out_mask = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.mask

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    stream_thread = LiveStreamThread(imgsz)
    stream_thread.daemon = True
    stream_thread.start()

    stream_fps = stream_thread.fps
    frame_spacing_thread = FrameSpacingThread(stream_fps)
    frame_spacing_thread.daemon = True
    frame_spacing_thread.start()

    frame_count = 0

    stream_width, stream_height = stream_thread.size
    camera = pyfakewebcam.FakeWebcam('/dev/video0', stream_width, stream_height)

    while True:
        img, im0s = frames_spaced.get(block=True)
        # print(img.shape)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        # TODO: eliminar si mask
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = 'livestream', '', im0s

            save_path = str(Path(out) / Path(p).name) + '%05d.jpg' % frame_count
            txt_path = str(Path(out) / Path(p).stem)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        if out_mask:
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            cv2.rectangle(im0, c1, c2, (255, 255, 255), thickness=-1)
                        else:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            frame_count += 1

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if True:
                    im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(save_path, im0)
                    camera.schedule_frame(im0_rgb)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
