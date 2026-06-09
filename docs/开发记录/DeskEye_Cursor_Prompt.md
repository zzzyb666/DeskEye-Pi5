# DeskEye 智能桌面专注助手 —— Cursor 工程提示词

## 0. 项目概述与目标系统

**目标系统**：DeskEye 是一个基于 Raspberry Pi 5 + CSI 摄像头的本地智能桌面监控系统，部署在用户的桌面正前方，实现以下核心功能：

- **座位检测**：自动检测用户是否在工作台前
- **专注度分析**：通过人脸朝向判断用户是否在专注工作
- **分心提醒**：检测手机等干扰物进入视野并记录
- **数据统计**：记录每日专注时长、分心次数，提供本地 Web 仪表盘查看

**硬件环境**（固定不可变）：
- Raspberry Pi 5 Model B Rev 1.0 (aarch64, 4×Cortex-A76 @ 1.5GHz)
- 4GB+ RAM
- CSI 金手指摄像头 (libcamera 协议栈，启动命令 `rpicam-hello -t 0`)
- Kernel 6.12, ARM Freq 1500MHz
- 无其他外设、无 Coral TPU、无网络依赖要求（完全离线运行）

**技术栈约束**（已验证可行）：
- Python 3.11+ (系统自带)
- OpenCV 4.x (通过 GStreamer pipeline 访问摄像头：`libcamerasrc ! ... ! appsink`)
- ONNX Runtime + ARM NN 后端 (推理引擎，禁止直接使用 PyTorch)
- SQLite (本地数据存储)
- Flask (Web 仪表盘)
- 模型格式：ONNX INT8 量化模型（禁止 FP32，慢 3.2×）

---

## 1. 逐步推进的技术路线（7个阶段）

每个阶段必须**独立可验证**，完成后更新项目记录文档 `PROJECT_LOG.md`。

### 阶段 0：项目脚手架 + 摄像头捕获验证
**目标**：建立项目目录结构，确认摄像头能正常捕获画面，OpenCV 环境就绪。

**交付物**：
```
deskeye/
├── PROJECT_LOG.md          # 项目记录文档（必须持续更新）
├── config.py               # 全局配置（摄像头参数、路径、阈值）
├── requirements.txt        # 依赖清单
├── src/
│   ├── __init__.py
│   ├── camera.py           # 摄像头封装类 (GStreamer pipeline)
│   └── utils.py            # 通用工具函数
├── models/                 # 存放 .onnx 模型文件
├── data/                   # SQLite 数据库
├── web/                    # Flask Web 仪表盘
└── tests/                  # 验证脚本
    └── test_camera.py      # 摄像头捕获测试
```

**验证标准**：运行 `tests/test_camera.py` 能弹出/保存一帧清晰的 640×480 画面。

---

### 阶段 1：运动检测 + 人物存在检测
**目标**：实现背景减除法检测画面变化，判断人物是否存在。

**技术方案**：OpenCV BackgroundSubtractorMOG2 + 轮廓检测，不依赖深度学习。

**交付物**：
- `src/motion_detector.py` — 运动检测模块
- `tests/test_motion.py` — 验证脚本

**验证标准**：人在画面前走动时控制台输出 "PRESENT"，离开 5 秒后输出 "ABSENT"。

---

### 阶段 2：人脸检测 + 朝向估计
**目标**：检测人脸位置，判断人脸朝向（正面/左/右/无人）。

**技术方案**：OpenCV DNN Face Detector (res10_300x300_ssd_iter_140000.caffemodel + deploy.prototxt)，或轻量级 ONNX 人脸检测模型（如 YuNet ONNX 版）。

**交付物**：
- `src/face_detector.py` — 人脸检测与朝向估计
- 自动下载人脸检测模型脚本

**验证标准**：运行测试脚本，画面中有人脸时画出 bbox 并标注朝向（"front"/"left"/"right"）。

---

### 阶段 3：手机/干扰物检测
**目标**：检测手机是否出现在画面中。

**技术方案**：YOLOv5n ONNX INT8 模型，ONNX Runtime ARM NN 后端推理。只关注 "cell phone" 类别。

**交付物**：
- `src/object_detector.py` — 物体检测模块
- 自动下载 yolov5n-int8.onnx 脚本

**验证标准**：手机出现在摄像头前时，控制台输出 "PHONE_DETECTED" 及置信度。

---

### 阶段 4：专注度评分 + 数据持久化
**目标**：综合座位状态、人脸朝向、手机检测结果，计算专注度评分并记录到 SQLite。

**专注度算法 v1**（先实现简单版本）：
- 座位有人 + 人脸正面 → +1 分/秒
- 座位有人 + 人脸侧转 → 0 分/秒
- 检测到手机 → -2 分/秒（最低 0）
- 座位无人 → 不加分

**交付物**：
- `src/focus_scorer.py` — 专注度评分模块
- `src/database.py` — SQLite 数据操作

**验证标准**：运行 1 分钟，数据库中能看到正确的时间序列记录和分数变化。

---

### 阶段 5：Web 仪表盘
**目标**：Flask 提供本地 Web 界面，查看今日专注统计。

**交付物**：
- `web/app.py` — Flask 应用
- `web/templates/index.html` — 仪表盘页面（纯 HTML+JS，无需框架）

**验证标准**：浏览器访问 `http://<pi-ip>:5000` 能看到今日专注时长、分心次数、分数趋势图。

---

### 阶段 6：系统集成 + 部署
**目标**：整合所有模块为守护进程，开机自启。

**交付物**：
- `main.py` — 主程序入口
- `deskeye.service` — systemd 服务文件
- 安装脚本 `install.sh`

**验证标准**：`sudo systemctl start deskeye` 后系统正常运行，重启后自动启动。

---

## 2. 真实工程问题（预期会遇到的）

开发过程中以下问题**极大概率会出现**，Cursor 必须优先解决这些问题：

| 问题 | 根因 | 优先解决方案 |
|------|------|-------------|
| 摄像头无法打开 | GStreamer pipeline 格式错误 / libcamera 权限 | 检查 `libcamerasrc` pipeline 格式，确认用户在 `video` 组 |
| 推理延迟 >500ms | 模型未量化 / 使用了 FP32 / 未开 ARM NN backend | 确认模型为 INT8，ONNX Runtime 使用 `providers=['CPUExecutionProvider']` with ARM NN |
| 低光环境下检测失效 | 摄像头自动曝光不足 | 通过 libcamera 控制参数调整曝光增益 |
| 背景减除误报 | 风扇/窗帘晃动 / 光线变化 | 调整 MOG2 `varThreshold`，添加形态学开运算去噪 |
| 长时间运行内存增长 | 帧缓冲未释放 / SQLite 连接未关闭 | 每 1000 帧强制 `gc.collect()`，使用连接池 |
| 人脸检测在侧脸时丢失 | SSD 人脸检测器对侧脸召回率低 | 降低置信度阈值到 0.3，结合运动检测兜底 |
| thermal throttle 降频 | CPU 持续高负载导致 >70°C | 添加散热片，设置 `scaling_governor=performance` |

---

## 3. 核心开发规范（必须严格遵守）

### 规范 A：每次生成代码前的强制步骤

在**每次**生成或修改代码前，必须按以下顺序执行：

1. **阅读当前目录下的文件**：列出 `src/`、`web/`、`tests/` 目录下已有文件，阅读关键文件内容
2. **判断项目开发进度**：根据 `PROJECT_LOG.md` 和已有代码，判断当前处于哪个阶段
3. **说明下一步应该做什么**：明确写出"当前处于阶段 X，下一步要实现 Y，需要修改/创建这些文件..."
4. **等待用户确认后再继续**：在获得用户明确回复（如"继续"、"确认"、"同意"）前，不生成任何代码

### 规范 B：文件操作原则

- **优先修改已有文件**，通过 `edit_file` 工具做精确替换
- **禁止随意删除重建**已有文件，除非文件内容完全错误且无法修复
- 新增文件仅在确实需要新模块时创建
- 每个文件修改后，说明改了什么、为什么改

### 规范 C：问题收敛流程

当遇到任何阻塞性问题时（代码跑不通、依赖装不上、逻辑有矛盾），**必须先执行以下收敛步骤，再写修复代码**：

1. **当前阶段卡在哪里**：一句话描述阻塞点
2. **这个问题的根因是什么**：分析根本原因，不是表面现象
3. **下一步最优先该做什么**：只选一个最优先的动作
4. **有哪些可选方案**：列出 2-3 个备选方案及各自的取舍

### 规范 D：任务拆分与验证

每个阶段的任务必须拆成**可验证的最小步骤**：
- 先做什么（能独立运行验证）
- 再做什么（依赖前一步但不扩大范围）
- 每完成一小步，说明验证方法（如"运行 `python tests/test_xxx.py` 应看到输出 `...`"）

### 规范 E：项目记录文档（PROJECT_LOG.md）

**每完成一个阶段或一次重要修改后，必须立即更新 `PROJECT_LOG.md`**，格式如下：

```markdown
# DeskEye 项目记录

## 当前阶段：阶段 X - [阶段名称]

## 已完成
- [日期] [时间] 完成了 [具体内容]
- [日期] [时间] 修改了 [文件]，原因是 [原因]

## 已知问题
- [问题描述] - [状态：已解决/未解决/绕过]

## 下一步
- [ ] 需要完成 [具体任务]
- [ ] 验证方法：[验证步骤]

## 上下文备忘
- [任何需要后续记住的关键信息]
```

更新 PROJECT_LOG.md 后，在回复中总结：**"刚刚改了什么 → 现在做到哪里了 → 下一步应该是什么"**。

---

## 4. 项目记录模板

以下是 `PROJECT_LOG.md` 的初始内容，复制到项目根目录：

```markdown
# DeskEye 项目记录

## 当前阶段：阶段 0 - 项目脚手架 + 摄像头捕获验证

## 已完成
- [待填写]

## 已知问题
- [待填写]

## 下一步
- [ ] 创建项目目录结构
- [ ] 编写 config.py 全局配置
- [ ] 编写 camera.py 摄像头捕获类
- [ ] 运行 test_camera.py 验证画面捕获

## 上下文备忘
- 硬件：Raspberry Pi 5 Model B, aarch64, 4×Cortex-A76 @ 1.5GHz
- 摄像头：CSI 金手指，libcamera 协议栈
- 技术栈：OpenCV + ONNX Runtime (ARM NN) + SQLite + Flask
- 模型约束：必须使用 INT8 量化 ONNX，禁止 PyTorch 直接推理
```

---

## 5. 关键代码约束

### 摄像头 GStreamer Pipeline（必须使用）
```python
# camera.py 中的 pipeline 模板
pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, framerate=15/1, format=NV12 ! "
    "queue ! videoconvert ! videoscale ! "
    "video/x-raw, width=640, height=480, format=BGR ! "
    "appsink drop=true"
)
```

### ONNX Runtime 推理配置（必须使用）
```python
import onnxruntime as ort

# 优先尝试 ARM NN，回退到 CPU
providers = ort.get_available_providers()
session = ort.InferenceSession(
    "model.onnx",
    providers=providers  # 自动选择最优 backend
)
```

### 数据库表结构
```sql
-- focus_events 表
CREATE TABLE IF NOT EXISTS focus_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT,        -- 'present', 'absent', 'focused', 'distracted', 'phone_detected'
    confidence REAL,        -- 置信度 0.0-1.0
    score_delta INTEGER,    -- 分数变化
    notes TEXT              -- 额外备注
);

-- daily_stats 表
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    total_focus_seconds INTEGER DEFAULT 0,
    distraction_count INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0,
    phone_detect_count INTEGER DEFAULT 0
);
```

---

## 6. 目录结构与文件清单（阶段 0 需要创建）

```
deskeye/
├── PROJECT_LOG.md              # [阶段0] 项目记录文档
├── config.py                   # [阶段0] 全局配置
├── requirements.txt            # [阶段0] Python 依赖
├── install.sh                  # [阶段6] 一键安装脚本
├── deskeye.service             # [阶段6] systemd 服务配置
├── main.py                     # [阶段6] 程序入口
├── models/
│   └── .gitkeep                # [阶段0] 模型存放目录
├── data/
│   └── .gitkeep                # [阶段0] 数据库目录
├── src/
│   ├── __init__.py             # [阶段0]
│   ├── camera.py               # [阶段0] 摄像头封装
│   ├── utils.py                # [阶段0] 工具函数
│   ├── motion_detector.py      # [阶段1] 运动检测
│   ├── face_detector.py        # [阶段2] 人脸检测
│   ├── object_detector.py      # [阶段3] 物体检测
│   ├── focus_scorer.py         # [阶段4] 专注度评分
│   └── database.py             # [阶段4] 数据库操作
├── web/
│   ├── app.py                  # [阶段5] Flask 应用
│   └── templates/
│       └── index.html          # [阶段5] 仪表盘
└── tests/
    ├── test_camera.py          # [阶段0] 摄像头测试
    ├── test_motion.py          # [阶段1] 运动检测测试
    ├── test_face.py            # [阶段2] 人脸检测测试
    ├── test_object.py          # [阶段3] 物体检测测试
    └── test_focus.py           # [阶段4] 专注度测试
```

---

## 7. 用户确认清单

在开始编码前，请用户确认以下事项（Cursor 在初始化时询问）：

1. **项目目标确认**：DeskEye 智能桌面专注助手是否符合需求？如有调整请说明。
2. **阶段优先级**：是否需要调整阶段的顺序或范围？
3. **安装环境**：树莓派上是否已安装 `python3-opencv` 和 `python3-pip`？
4. **网络访问**：树莓派能否访问互联网以下载 ONNX 模型？
5. **存储位置**：项目代码存放在哪个目录？（默认 `~/deskeye/`）

---

## 8. 第一条消息模板（Cursor 初始化时发送）

```
你好！我是你的 DeskEye 项目开发助手。

我正在阅读项目目录和 PROJECT_LOG.md 来了解当前进度...

[列出当前目录结构和已有文件]

根据项目记录，当前处于【阶段 X - XXX】：
- 已完成：[列出已完成项]
- 下一步：[描述下一步任务]

在继续之前，我需要确认：
1. 上述进度判断是否准确？
2. 是否有需要优先处理的阻塞问题？
3. 你是否同意我按"下一步"开始编写代码？

请确认后我继续。
```

---

## 9. 阶段完成消息模板

每个阶段完成后，Cursor 必须发送如下格式的总结：

```
✅ 【阶段 X - XXX】已完成

刚刚改了什么：
- 创建了 [文件A]，实现了 [功能]
- 修改了 [文件B]，修复了 [问题]
- 更新了 PROJECT_LOG.md

现在做到哪里了：
- [当前整体进度描述]
- 所有已有功能验证状态：[通过/未验证]

下一步应该是什么：
- 阶段 X+1：[描述]
- 需要先完成：[子任务列表]
- 验证方法：[如何验证]

是否继续进入下一阶段？请确认。
```

---

## 10. 必须记住的核心原则

1. **先问后做**：每次行动前说明计划，等用户确认
2. **小步快跑**：每个改动可独立验证，不堆积大功能
3. **文档先行**：改代码前先看文件，改完后更新 PROJECT_LOG.md
4. **问题收敛**：卡顿时先分析根因和选项，再写修复代码
5. **Pi 5 专用**：所有代码必须适配 aarch64 + libcamera + ARM NN
6. **离线优先**：不依赖云服务，所有推理本地完成
