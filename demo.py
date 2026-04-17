import streamlit as st
import cv2
import tempfile
import imageio
from ultralytics import YOLO

# ================= 1. 网页全局配置 =================
st.set_page_config(page_title="智能防溺水监控系统", page_icon="🚨", layout="wide")
st.title("🌊 泳池智能防溺水监控与预警系统")
st.markdown("系统说明：上传监控视频后，点击【开始智能分析】。系统会自动处理并在人员消失时**锁定最后位置并触发红色警报**。")

# ================= 2. 加载模型 =================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ================= 3. 界面布局 =================
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ 操作面板")
    uploaded_file = st.file_uploader("📂 请上传测试视频 (.mp4)", type=['mp4', 'avi', 'mov'])
    
    # 状态提示区域
    status_msg = st.empty()
    progress_bar = st.empty()
    
with col1:
    st.subheader("📺 智能分析播放器")
    video_player = st.empty()
    video_player.info("等待上传并分析视频...")

# ================= 4. 离线渲染核心逻辑 =================
if uploaded_file is not None:
    # 放置一个分析按钮，用户点击后才开始
    with col2:
        start_btn = st.button("🚀 开始智能分析", use_container_width=True)

    if start_btn:
        status_msg.info("⏳ 正在读取视频文件...")
        
        # 将上传的视频保存到临时文件
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        tfile_in.close()
        
        # 准备输出视频的临时文件
        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out_path = tfile_out.name
        tfile_out.close()

        # 使用 imageio 读取视频 (保证跨平台兼容性)
        reader = imageio.get_reader(tfile_in.name, 'ffmpeg')
        fps = reader.get_meta_data().get('fps', 25)
        
        # 使用 imageio 写入视频 (编码为网页支持的 h264 格式)
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None)

        last_known_bbox = None
        frame_count = 0
        
        status_msg.warning("⚙️ AI 正在逐帧分析并渲染视频，请稍候...")
        # 进度条
        p_bar = progress_bar.progress(0.0)

        # 逐帧处理并画框
        for frame_rgb in reader:
            frame_count += 1
            
            # 缩放加快处理速度
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            
            # YOLO 需要预测 RGB 格式即可
            results = model.predict(frame_rgb, classes=[0], verbose=False)
            
            boxes = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # 转成 BGR 供 OpenCV 画图使用
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # --------- 剧情判定逻辑 ---------
            if len(boxes) > 0:
                # 有人：绿色框
                last_known_bbox = boxes[0]
                x1, y1, x2, y2 = map(int, last_known_bbox)
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_bgr, "SWIMMING", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 画面左上角打上监控状态（增强科技感）
                cv2.putText(frame_bgr, "SYS STATUS: NORMAL", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            else:
                # 没人：红色警报框定格
                if last_known_bbox is not None:
                    x1, y1, x2, y2 = map(int, last_known_bbox)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(frame_bgr, "DANGER!", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    cv2.putText(frame_bgr, "ALARM: DROWNING DETECTED!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 再转回 RGB 写入视频文件
            frame_final = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_final)
            
            # 让进度条动起来 (做个假的循环视觉效果，防止视频长度读取失败)
            p_bar.progress(min((frame_count % 100) / 100.0 + 0.01, 1.0))

        # 收尾工作
        writer.close()
        reader.close()
        p_bar.empty()
        status_msg.success("✅ 分析渲染完成！请在左侧播放器查看结果。")
        
        # 将生成的视频发送给网页的播放器
        with open(out_path, 'rb') as video_file:
            video_bytes = video_file.read()
            video_player.video(video_bytes)
