import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

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
col1, col2 = st.columns([2, 1]) # 左侧视频占 2/3，右侧状态占 1/3

with col1:
    st.subheader("📺 实时监控画面")
    # 视频播放占位符
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 系统状态面板")
    # 状态栏和日志占位符
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    # 初始状态
    status_placeholder.info("等待上传监控视频...")
    
    # 上传视频组件
    uploaded_file = st.file_uploader("请上传测试视频 (.mp4)", type=['mp4', 'avi', 'mov'])

# ================= 4. 核心“剧情”处理逻辑 =================
if uploaded_file is not None:
    # 每次上传新视频，初始化状态
    status_placeholder.success("✅ 视频已加载，正在初始化分析引擎...")
    
    # 将上传的视频保存为临时文件，供 OpenCV 读取
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # --- Demo 核心伪装变量 ---
    last_known_bbox = None  # 记录人最后出现的坐标
    is_drowning = False     # 溺水警报状态
    
    log_text = ""
    
    # 逐帧读取视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("视频播放结束。")
            break
            
        # 画面降采样加速运行
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO 检测 (只检测人 classes=[0])
        results = model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # --------- 逻辑判断开始 ---------
        if len(boxes) > 0:
            # 只要检测到了人
            is_drowning = False
            last_known_bbox = boxes[0] # 更新人的最后坐标 (取第一个人)
            
            x1, y1, x2, y2 = map(int, last_known_bbox)
            
            # 画绿色框 + 英文提示 (避开云端中文乱码问题)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "SWIMMING", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 网页 UI 显示中文状态
            status_placeholder.success("🟢 状态正常：人员正在游泳")
            
        else:
            # 没有检测到人！
            if last_known_bbox is not None:
                # 如果之前有人的坐标，说明人消失了 -> 触发沉底溺水警报！
                is_drowning = True
                
                x1, y1, x2, y2 = map(int, last_known_bbox)
                
                # 在人员消失的最后一刻位置，画红色警报框定格
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, "DANGER!", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # 网页 UI 报警
                status_placeholder.error("🚨 溺水警报！人员消失 (沉底)，请立即救援！")
                
            else:
                # 视频一开始就没人
                status_placeholder.info("👀 画面中暂时无人员...")
        
        # --------- 渲染与输出 ---------
        # 将 OpenCV 的 BGR 转为网页需要的 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()
