@echo off
chcp 65001 >nul
echo 启动 A股量化因子可视化服务...
echo 启动后请查看终端输出的访问地址（如 http://localhost:8002）
echo.
py -3 app.py
