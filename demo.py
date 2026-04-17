import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# ================= 1. 网页全局配置 =================
st.set_page_config(page_title="智能防溺水监控系统", page_icon="🚨", layout="wide")
st.title("🌊 泳池智能防溺水监控与预警系统")
st.markdown("系统说明：实时检测画面人员，**当人员突然在画面中消失（模拟沉底/溺水），系统将锁定最后位置并触发红色警报。**")

# ================= 2. 加载模型 (缓存机制加速) =================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ================= 3. 界面布局 (左右分栏) =================
col1, col2 = st.columns([2, 1]) 

with col1:
    st.subheader("📺 实时监控画面")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 系统状态面板")
    status_placeholder = st.empty()
    
    status_placeholder.info("等待上传监控视频...")
    uploaded_file = st.file_uploader("请上传测试视频 (.mp4)", type=['mp4', 'avi', 'mov'])

# ================= 4. 核心“剧情”处理逻辑 =================
if uploaded_file is not None:
    status_placeholder.success("✅ 视频已加载，正在初始化分析引擎...")
    
    # 将上传的视频保存为临时文件供 OpenCV 读取
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    last_known_bbox = None  # 记录人最后出现的坐标
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("视频播放结束。")
            break
            
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO 检测 (只检测人)
        results = model.predict(frame, classes=[0], verbose=False)
        
        # 处理可能返回None的情况以防止报错
        boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # --------- 逻辑判断开始 ---------
        if len(boxes) > 0:
            # 检测到了人
            last_known_bbox = boxes[0] # 更新人的最后坐标
            x1, y1, x2, y2 = map(int, last_known_bbox)
            
            # 绿框 + 英文
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "SWIMMING", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            status_placeholder.success("🟢 状态正常：人员正在游泳")
            
        else:
            # 没有检测到人
            if last_known_bbox is not None:
                # 之前有人，现在消失了 -> 触发定格溺水警报
                x1, y1, x2, y2 = map(int, last_known_bbox)
                
                # 停留在原地的红框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, "DANGER!", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                status_placeholder.error("🚨 溺水警报！人员消失 (沉底)，请立即救援！")
            else:
                status_placeholder.info("👀 画面中暂时无人员...")
        
        # 转换并显示到网页
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()
