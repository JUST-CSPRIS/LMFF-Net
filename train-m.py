import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 强制只使用显卡1
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from ultralytics import YOLO



if __name__ == '__main__':

    yaml_names = ["yolov5-repghost"]  # 定义一个包含模型配置文件名称的列表

    for yaml_name in yaml_names:  # 遍历列表中的每个模型配置文件名称
        yaml_path = f'/home/xie/xcl/paper/code/yolov8/ultralytics/ultralytics/cfg/models/v8/{yaml_name}.yaml'  # 构建模型配置文件的完整路径

        # 初始化模型
        model = YOLO(yaml_path)  # 使用配置文件初始化YOLO模型

        # 开始训练
        model.train(
            data='/home/xie/xxs/yolo11（小目标）/ultralytics/cfg/datasets/visdrone.yaml',  # 数据集配置文件路径
            cache=False,  # 是否使用缓存
            amp=False,

            imgsz=640,  # 输入图像大小
            epochs=300,  # 训练的总轮次
            batch=8,  # 批量大小
            close_mosaic=0,
            workers=4,  # 数据加载的线程数
            device='1',  # 指定使用的设备（显卡1）
            optimizer='SGD',  # 优化器类型
            project='runs/visdrone',  # 保存训练结果的项目目录
            name=f'{yaml_name}',  # 使用yaml名称作为训练任务名
            # 其他保持默认的参数
        )