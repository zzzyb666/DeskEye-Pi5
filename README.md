# DeskEye — 树莓派智能桌面专注助手

基于 **Raspberry Pi 5 + CSI 摄像头** 的本地离线桌面监控系统：检测是否在座、人脸朝向、手机干扰，计算专注度并写入 SQLite，通过浏览器查看统计。

> 硬件：Pi 5 (aarch64) + OV5647（libcamera）  
> 技术栈：Python 3.11+ · OpenCV · ONNX Runtime · SQLite · Flask

## 快速开始

**完整安装与运行步骤见 → [启动说明.md](启动说明.md)**

```bash
git clone <你的仓库地址> ~/deskeye
cd ~/deskeye
chmod +x install.sh
./install.sh
nano deskeye.env          # 填写摄像头路径
sudo systemctl start deskeye
```

浏览器打开：`http://<树莓派IP>:5000/`

## 项目结构（简览）

| 目录/文件 | 作用 |
|-----------|------|
| `main.py` | 生产入口：检测 + Web 双线程 |
| `src/` | 摄像头、检测、计分、数据库 |
| `web/` | Flask 仪表盘 |
| `tests/` | 分阶段验证脚本 |
| `install.sh` | 一键安装与 systemd 配置 |

## 开发记录

过程日志见 `PROJECT_LOG.md` 与 `docs/开发记录/`。

## 许可

学习用例项目，按需自用与修改。
