# 铁路智能监控系统

## 项目简介
基于计算机视觉的铁路场景智能监控系统，实现列车进出站分析、道口杂物检测、人员手势识别等功能。

## 项目结构
railway_monitoring/
├── weights/ # 模型权重文件
├── models/ # 模型定义
├── utils/ # 工具函数
├── config/ # 配置文件
├── src/ # 源代码
├── data/ # 数据文件
├── outputs/ # 输出结果
├── tests/ # 测试代码
├── docs/ # 文档
├── scripts/ # 工具脚本
└── notebooks/ # Jupyter笔记本


## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置相机参数：编辑 `config/cameras/` 下的配置文件
3. 运行系统：`python main.py --config config/system/main.yaml`

## 功能模块
- 列车进出站检测
- 道口杂物检测  
- 人员手势识别
- 人员/车辆停留检测
- 违规使用手机检测
