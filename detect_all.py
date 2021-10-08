import cv2
import numpy as np
import torch
from PIL import Image
import xlwt

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator


line_num = 0
circle_num = 0
min_lineLength = 20
min_lineGap = 10
roi_num = 0

def _generate_line_num():
    global line_num
    line_num += 1
    return line_num


def _generate_circle_num():
    global circle_num
    circle_num += 1
    return circle_num


def _generate_roi_num():
    global roi_num
    roi_num += 1
    return roi_num


# 完成直线
class Line:
    def __init__(self, start_coordinate, end_coordinate):
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate
        self.num = _generate_line_num()


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.num = _generate_circle_num()




class Roi:
    def __init__(self, top_left, bottle_right):
        self.top_left = top_left
        self.bottle_right = bottle_right
        self.num = _generate_roi_num()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
weights = r'work/weights/best.pt'
imgsz = 640
conf_thres = 0.4
iou_thres = 0.25


def load_model():
    model = attempt_load(weights, map_location=device)  # load FP32 model

    if half:
        model.half()  # to FP16

    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    return model


if __name__ == '__main__':

    # 字母和数字部分
    img_path = "test.jpg"
    model = load_model()
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    img0 = cv2.imread(img_path)
    src = img0
    pil_img = Image.open(img_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    if len(img.shape) == 3:
        img = img[None]
    img = img / 255.
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=300)
    # print(pred)
    aims = []
    for i, det in enumerate(pred):
        s = ''
        # 这里使用的是letterbox()改变后的img尺寸
        s += '%gx%g ' % img.shape[2:]
        # torch.tensor()使用的是原图的尺寸
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        annotator = Annotator(src, line_width=3, pil=not ascii)
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 第一次将下列代码的img0写成了img QAQ
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # save bbox
            for *xyxy, conf, cls in reversed(det):  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                aims.append(aim)

    s_width, s_height = pil_img.size[0], pil_img.size[1]

    rois = []
    if len(aims):
        for i, det in enumerate(aims):
            _, x_center, y_center, width, height = det

            x_center, width = s_width * float(x_center), s_width * float(width)
            y_center, height = s_height * float(y_center), s_height * float(height)
            # print(y_center, height)
            top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
            bottle_right = (int(x_center + width / 2.), int(y_center + height / 2.))

            roi = Roi(top_left, bottle_right)
            rois.append(roi)

            print(top_left, bottle_right)

            """b0 = (float(x_center) + 1) * s_width - s_width * float(width) / 2.
            b1 = (float(y_center) + 1) * s_height - s_height * float(height) / 2.
            b2 = (float(x_center) + 1) * s_width + s_width * float(width) / 2.
            b3 = (float(y_center) + 1) * s_height + s_height * float(height) / 2."""

            """b0 = float(x_center) * s_width + 1 - s_width * float(width) / 2.
            b1 = float(y_center) * s_height + 1 - s_height * float(height) / 2.
            b2 = float(x_center) * s_width + 1 + s_width * float(width) / 2.
            b3 = float(y_center) * s_height + 1 + s_height * float(height) / 2.
            top_left = (int(b0), int(b1))
            bottle_right = (int(b2), int(b3))


            print(top_left, bottle_right)"""
            cv2.rectangle(src, top_left, bottle_right, (0, 255, 0), 1)


    # print(rois)
    #         提取直线和圆
    linesarray = []
    circlesarray = []
    path = "test.jpg"
    # img = cv2.imread(path)
    # src = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("after", imgs)
    # cv2.waitKey()
    # src = cv2.GaussianBlur(src, (3, 3), 0)
    # src = cv2.GaussianBlur(imgs, (3, 3), 0)
    src2 = cv2.fastNlMeansDenoising(img0, 5)
    edges = cv2.Canny(src2, 50, 150, apertureSize=3)
    # cv2.imshow("after", edges)
    # cv2.waitKey()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30)
    # 利用HoughCircle函数提取，但是效果不是很理想
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=13, minRadius=0, maxRadius=10)
    for x, y, r in circles[0, :]:
        center = (int(x), int(y))
        r = int(r)
        circle = Circle((x, y), r)
        circlesarray.append(circle)
        cv2.circle(src, center, r, (0, 0, 255), 1)
        cv2.circle(src, center, 1, (0, 0, 255), 1)
    # print(circles)
    # print(lines)
    # 观察到所有的圆孔都是低阈值区域，可以设定域值进行阈值提取
    """src = cv2.fastNlMeansDenoising(src, 5)
    ret, thresh = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(
        image=edges,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_NONE,
        hierarchy=None,
        offset=None)
    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        if area > 1000 or area < 20:
            continue
        ratio = rect[2] / rect[3]
        if 0.9 < ratio < 1.1:
            circlesarray.append(contour)
        # cv2.drawContours(src, contour, 1, (0, 255, 0), 3)
    print(circlesarray)
    src = cv2.drawContours(imgs, circlesarray, -1, (0, 255, 0), 1)"""
    for x1, y1, x2, y2 in lines[:, 0]:
        start_coordinate = (x1, y1)
        end_coordinate = (x2, y2)
        line = Line(start_coordinate, end_coordinate)
        linesarray.append(line)
        cv2.line(src, start_coordinate, end_coordinate, (200, 200, 0), 1)

    # 将字母数字、直线和圆的位置相关参数写入xls文件
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet_line = workbook.add_sheet("sheet_line")
    sheet_line.write(0, 0, "编号")
    sheet_line.write(0, 1, "起点坐标")
    sheet_line.write(0, 2, "终点坐标")
    for line in linesarray:
        # print(line.num)
        sheet_line.write(int(line.num), 0, str(line.num))
        sheet_line.write(int(line.num), 1, str(line.start_coordinate))
        sheet_line.write(int(line.num), 2, str(line.end_coordinate))
    sheet_circle = workbook.add_sheet("sheet_circle")
    sheet_circle.write(0, 0, "编号")
    sheet_circle.write(0, 1, "圆心坐标")
    sheet_circle.write(0, 2, "半径")
    for circle in circlesarray:
        sheet_circle.write(int(circle.num), 0, str(circle.num))
        sheet_circle.write(int(circle.num), 1, str(circle.center))
        sheet_circle.write(int(circle.num), 2, str(circle.radius))
    sheet_roi = workbook.add_sheet("sheet_roi")
    sheet_roi.write(0, 0, "编号")
    sheet_roi.write(0, 1, "左上角坐标")
    sheet_roi.write(0, 2, "右下角坐标")
    for roi in rois:
        sheet_roi.write(int(roi.num), 0, str(roi.num))
        sheet_roi.write(int(roi.num), 1, str(roi.top_left))
        sheet_roi.write(int(roi.num), 2, str(roi.bottle_right))
    workbook.save("提取结果.xls")
    # cv2.imwrite("after.jpg", src)
    cv2.imshow("after.jpg", src)
    cv2.waitKey()
