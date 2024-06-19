import argparse
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(777)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


dict_answer = {0: 'TP', 1: 'FN', 2: 'FP'}  # 'TN' does not exists for object detection.
const_TP = 0
const_FN = 1
const_FP = 2


# def write_log(filename, target_class, target_coord, pred_class, pred_coord, conf, ans_iou, c_TP, c_FP, c_P, c_R):
def write_log(filename, target_class, target_coord, pred_class, pred_coord, conf, ans_iou):
    row = {'FILENAME': filename,
           'target_class': target_class,
           'target_xywh': target_coord,
           'pred_class': pred_class,
           'pred_xywh': pred_coord,
           'confidence': conf,
           'ans_iou': dict_answer[ans_iou],
           # 'c_TP' : c_TP,
           # 'c_FP' : c_FP,
           # 'c_P' : c_P,
           # 'c_R' : c_R
           }
    return row


def get_p_r_data(df, cls, n_target):
    df = df.loc[df['pred_class'] == cls]
    n_tp = np.repeat(0, len(df))
    n_fp = np.repeat(0, len(df))
    c_p = np.repeat(0, len(df))
    c_r = np.repeat(0, len(df))

    i = 0
    for index, row in df.iterrows():
        if row['ans_iou'] == 'TP':
            n_tp[i:] += 1
        if row['ans_iou'] == 'FP':
            n_fp[i:] += 1
        # if not np.isnan(row['target_class']):
        #   n_target[i:] += 1
        i += 1

    c_p = n_tp / (n_tp + n_fp)
    c_r = n_tp / n_target

    return n_tp, n_fp, c_p, c_r


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         save_ans_log=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = Darknet(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # 새로 추가된 초기값
    list_row = []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)

            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:  # if tbox has no matching pbox (FN)
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

                    # get tbox_pbox logs
                    if save_ans_log:
                        gn = np.tile(shapes[si][0][::-1], 2)
                        tbox_original = xywh2xyxy(labels.clone()[:, 1:5]) * whwh
                        tbox_original = scale_coords(img[si].shape[1:], tbox_original, shapes[si][0], shapes[si][1])
                        tbox_original = (xyxy2xywh(tbox_original) / gn).tolist()
                        tbox_class = tcls

                        # write log by imagefile - target box
                        for k in range(0, nl):
                            row = write_log(paths[0], tbox_class[k], tbox_original[k], np.nan, np.nan, np.nan, const_FN)

                            list_row.append(pd.DataFrame([row]))

                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # pbox matched tbox index
            list_pbox_matched_tbox = [-1 for i in range(len(pred))]

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d.item())

                                list_pbox_matched_tbox[pi[j]] = d

                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # get tbox_pbox logs
            if save_ans_log:
                gn = np.tile(shapes[si][0][::-1], 2)
                tbox_original = tbox.clone()[:, :4]
                tbox_original = scale_coords(img[si].shape[1:], tbox_original, shapes[si][0], shapes[si][1])
                tbox_original = (xyxy2xywh(tbox_original) / gn).tolist()
                tbox_class = tcls

                pbox_original = pred.clone()
                pbox_original = scale_coords(img[si].shape[1:], pbox_original[:, :4], shapes[si][0], shapes[si][1])
                pbox_original = (xyxy2xywh(pbox_original) / gn).tolist()
                pbox_class = pred[:, 5].tolist()
                pbox_conf = pred[:, 4].tolist()

                list_correct_iou50 = correct[:, 0].tolist()

                # write log by imagefile - target box
                for k in range(0, nl):

                    # if k in detected:  # if tbox has a matching pbox at IOU : 0.5 (TP)
                    if k in list_pbox_matched_tbox:  # if tbox has a matching pbox at IOU : 0.5 (TP)
                        pbox_index = list_pbox_matched_tbox.index(k)

                        if list_correct_iou50[pbox_index] == True:
                            row = write_log(paths[0], tbox_class[k], tbox_original[k],
                                            pbox_class[pbox_index], pbox_original[pbox_index],
                                            pbox_conf[pbox_index],
                                            const_TP)
                        else:  # if pbox has not enough iou (FP)
                            row = write_log(paths[0], tbox_class[k], tbox_original[k],
                                            pbox_class[pbox_index], pbox_original[pbox_index],
                                            pbox_conf[pbox_index],
                                            const_FP)
                    else:  # if tbox has no matching pbox (FN)
                        row = write_log(paths[0], tbox_class[k], tbox_original[k], np.nan, np.nan, np.nan, const_FN)

                    list_row.append(pd.DataFrame([row]))

                # if pbox has not matching tbox (FP)
                for pbox_index in [i for i, v in enumerate(list_pbox_matched_tbox) if v == -1]:
                    row = write_log(paths[0], np.nan, np.nan,
                                    pbox_class[pbox_index], pbox_original[pbox_index], pbox_conf[pbox_index],
                                    const_FP)
                    list_row.append(pd.DataFrame([row]))



        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions
            #plot_images(img, output_to_target(output.cpu().numpy(), width, height), paths, f, names)  # predictions


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # W&B logging
    if plots and wandb:
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if save_ans_log:
        df = pd.concat(list_row)
        df['target_xywh'] = df.apply(lambda x: np.round(x['target_xywh'], 5), axis=1)
        df['pred_xywh'] = df.apply(lambda x: np.round(x['pred_xywh'], 5), axis=1)
        df['FILENAME'] = df['FILENAME'].str.extract(r'([^\\]+$)')

        n_target_by_cls = df['target_class'].value_counts().sort_index()

        list_df = []
        list_ap = []
        for i, cls in enumerate(n_target_by_cls.index):
            df_tmp = df.loc[df['pred_class'] == cls]
            df_tmp = df_tmp.sort_values('confidence', ascending=False)

            n_tp, n_fp, c_p, c_r = get_p_r_data(df_tmp, cls, n_target_by_cls[cls])

            df_tmp['n_tp'] = n_tp
            df_tmp['n_fp'] = n_fp
            df_tmp['c_p'] = c_p
            df_tmp['c_r'] = c_r

            # calculate ap50
            ap, mpre, mrec = compute_ap(c_r, c_p)
            print('class :  {0} > target : {1}  /  AP@0.5 : {2:.5f}'.format(i, n_target_by_cls[cls], ap))
            list_ap.append(ap)
            list_df.append(df_tmp)

            # plot p-r curve
            px, py = np.linspace(0, 1, 1000), []  # for plotting
            py.append(np.interp(px, mrec, mpre))

            fname = 'precision-recall_curve_class_{0}.png'.format(i)

            # py = np.stack(py, axis=1)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(px, py[0], linewidth=0.5, color='grey')  # plot(recall, precision)
            ax.plot(px, py[0], linewidth=2, color='blue', label='class {0} AP@0.5 {1:.4f}'.format(i, ap))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.legend()
            fig.tight_layout()
            fig.savefig(save_dir / fname, dpi=200)

        print('class :  all > target : {0}  /  AP@0.5 : {1:.5f}'.format(np.sum(n_target_by_cls), np.mean(list_ap)))
        df_export = pd.concat(list_df)
        df_export = df_export.reset_index(drop=True)
        df_export = df_export.reset_index(drop=False)
        df_export.to_csv(save_dir / 'test_log.csv', index=False)


    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', default=True, help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', default=False, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', default=False, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--save-ans-log', action='store_true', default=False, help='save a answer log by bounding boxes')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally

        now = datetime.now()
        print("Model Test Start at ", now)
        print('\n')

        if opt.save_ans_log:
            opt.batch_sie = 1

        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.save_ans_log,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )

        now = datetime.now()
        print('\n')
        print("Model Test End at ", now)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolor_p6.pt', 'yolor_w6.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot

