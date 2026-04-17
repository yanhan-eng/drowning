import streamlit as st
import cv2
import tempfile
import imageio
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import os
import numpy as np

# ================= 1. 网页全局配置 =================
st.set_page_config(page_title="防溺水监控预警系统", page_icon="🚨", layout="wide")
st.title("🌊 泳池智能防溺水监控与预警系统")
st.markdown("系统说明：上传视频后点击【开始智能分析】。人员突然消失时（模拟沉底），系统将锁定最后位置并触发红色预警。")

# ================= 2. 核心资源加载 (模型与中文字体) =================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

@st.cache_resource
def get_chinese_font():
    """在云端自动下载并加载中文字体，解决 OpenCV 无法写中文的问题"""
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        # 从可靠的开源仓库下载黑体字体
        font_url = "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf"
        try:
            urllib.request.urlretrieve(font_url, font_path)
        except:
            st.error("字体下载失败，请检查云端网络环境。")
            return ImageFont.load_default()
    return ImageFont.truetype(font_path, 30)

model = load_model()
chinese_font = get_chinese_font()

# ================= 3. 界面布局 =================
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ 操作面板")
    uploaded_file = st.file_uploader("📂 请上传测试视频 (.mp4)", type=['mp4', 'avi', 'mov'])
    
    status_msg = st.empty()
    progress_bar = st.empty()
    
with col1:
    st.subheader("📺 智能分析播放器")
    video_player = st.empty()
    video_player.info("等待上传并分析视频...")

# ================= 4. 离线极速渲染逻辑 =================
if uploaded_file is not None:
    with col2:
        start_btn = st.button("🚀 开始极速智能分析", use_container_width=True)

    if start_btn:
        status_msg.info("⏳ 正在读取视频文件...")
        
        # 保存临时文件
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        tfile_in.close()
        
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out_path = tfile_out.name
        tfile_out.close()

        # 读取视频元数据
        reader = imageio.get_reader(tfile_in.name, 'ffmpeg')
        fps = reader.get_meta_data().get('fps', 25)
        total_frames = reader.count_frames()
        if total_frames == float('inf') or total_frames == 0:
            total_frames = 150 # 默认给个假进度条基数
            
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None)

        last_known_bbox = None
        current_boxes = []
        
        status_msg.warning("⚡ 游泳监测分析中，请稍候...")
        p_bar = progress_bar.progress(0.0)

        # 核心优化 1：设置抽帧频率 (每处理1帧，跳过2帧，速度提升3倍！)
        process_every_n_frames = 3 

        for i, frame in enumerate(reader):
            # imageio 读取的是 RGB 格式
            frame_rgb = np.array(frame)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            
            # --- AI 分析 (抽帧加速 + 降低推理分辨率 imgsz=320 提速) ---
            if i % process_every_n_frames == 0:
                results = model.predict(frame_rgb, classes=[0], imgsz=320, verbose=False)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    current_boxes = results[0].boxes.xyxy.cpu().numpy()
                else:
                    current_boxes = []

            # --- 剧情判断与画框 (使用 RGB 颜色) ---
            if len(current_boxes) > 0:
                # 状态：游泳中
                last_known_bbox = current_boxes[0]
                x1, y1, x2, y2 = map(int, last_known_bbox)
                
                # 画绿框 (RGB: 0, 255, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                text = "✅ 正常游泳"
                text_color = (0, 255, 0)
                sys_text = "系统状态: 监控正常"
                
            else:
                # 状态：人员消失 (模拟溺水)
                if last_known_bbox is not None:
                    x1, y1, x2, y2 = map(int, last_known_bbox)
                    
                    # 定格画红框 (RGB: 255, 0, 0)
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    
                    text = "🚨 溺水 (沉底)！"
                    text_color = (255, 0, 0)
                    sys_text = "警报: 发现溺水人员！"
                else:
                    # 一开始就没人
                    text = ""
                    sys_text = "系统状态: 画面无目标"
                    text_color = (200, 200, 200)

            # --- 核心优化 2：使用 PIL 绘制中文 ---
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            # 绘制左上角全局状态
            draw.text((15, 15), sys_text, font=chinese_font, fill=text_color)
            
            # 绘制人头顶的状态
            if text and last_known_bbox is not None:
                x1, y1 = map(int, last_known_bbox[:2])
                # 画个黑色阴影防止看不清
                draw.text((x1 + 2, max(0, y1 - 38)), text, font=chinese_font, fill=(0, 0, 0))
                draw.text((x1, max(0, y1 - 40)), text, font=chinese_font, fill=text_color)
                
            # 转回 NumPy 数组写入视频
            frame_final = np.array(pil_img)
            writer.append_data(frame_final)
            
            # 更新进度条
            progress = min(i / total_frames, 1.0)
            p_bar.progress(progress)

        # 释放资源
        writer.close()
        reader.close()
        p_bar.empty()
        status_msg.success("✅ 分析完成！请点击左侧播放器播放。")
        
        # 在网页播放最终带中文特效的 MP4
        with open(out_path, 'rb') as video_file:
            video_player.video(video_file.read())
