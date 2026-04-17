import cv2
import math
import tkinter as tk
from tkinter import font, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import winsound  # 用于在 Windows 上发出警报声音


class MockDrowningDemo:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")

        print("正在加载 YOLO 模型...")
        self.model = YOLO('yolov8n.pt')

        self.vid = None
        self.playing_video = False

        # --- 剧情演示逻辑变量 ---
        # 记录每个检测目标的存活帧数 { 内部id: {'frames': 存活帧数, 'pos': (cx, cy)} }
        self.hidden_tracks = {}
        self.next_internal_id = 0

        # 触发参数 (假设视频处理速度约 25-30 FPS)
        self.drowning_threshold_frames = 75  # 约 2.5 ~ 3 秒后强制变为溺水
        self.global_alarm = False  # 全局警报状态
        self.alarm_sound_counter = 0  # 控制警报声音频率，防止声音卡顿

        # ================= GUI 布局 =================
        main_frame = tk.Frame(window, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 左侧：视频显示区 ---
        left_frame = tk.Frame(main_frame, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_width = 720
        self.canvas_height = 480
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="#222222")
        self.canvas.pack(padx=10, pady=10, anchor=tk.CENTER)

        # --- 右侧：状态检测控制区 ---
        right_frame = tk.Frame(main_frame, width=250, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.custom_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.status_label = tk.Label(right_frame, text="等待加载视频", font=self.custom_font, fg="white", bg="gray",
                                     pady=10)
        self.status_label.pack(fill=tk.X, pady=(10, 20))

        list_label = tk.Label(right_frame, text="👥 画面人员状态：", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        list_label.pack(anchor=tk.W)

        self.log_text = tk.Text(right_frame, width=25, height=15, font=("Helvetica", 11), state=tk.DISABLED, bg="white")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        btn_font = ("Helvetica", 12)
        tk.Button(right_frame, text="🎞️ 加载演示视频", command=self.load_video, font=btn_font, height=2,
                  bg="#e0e0e0").pack(fill=tk.X, pady=10)
        tk.Button(right_frame, text="❌ 退出系统", command=self.close_app, font=btn_font, height=2, bg="#e0e0e0").pack(
            fill=tk.X, pady=5)

        self.window.protocol("WM_DELETE_WINDOW", self.close_app)
        self.window.mainloop()

    def get_chinese_font(self):
        try:
            return ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
        except:
            try:
                return ImageFont.truetype("simhei.ttf", 20)  # 黑体
            except:
                return ImageFont.load_default()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.playing_video = False
            if self.vid is not None:
                self.vid.release()

            self.vid = cv2.VideoCapture(file_path)
            self.hidden_tracks.clear()
            self.next_internal_id = 0
            self.playing_video = True
            self.update_video_frame()

    def play_alarm_sound(self):
        """异步播放系统警报声（仅限 Windows），不会卡顿画面"""
        try:
            # 使用 Windows 默认的感叹号声音，SND_ASYNC 表示后台播放不阻塞
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except:
            pass

    def update_video_frame(self):
        if self.playing_video and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

                # YOLO 预测 (仅框出人)
                results = self.model.predict(frame, classes=[0], verbose=False)
                self.global_alarm = False

                current_detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        current_detections.append((cx, cy, x1, y1, x2, y2))

                new_hidden_tracks = {}
                texts_to_draw = []
                person_statuses = []

                # 后台悄悄匹配框（用于计帧，不显示ID）
                for det in current_detections:
                    cx, cy, x1, y1, x2, y2 = det
                    best_match_id = None
                    min_dist = float('inf')

                    for track_id, data in self.hidden_tracks.items():
                        last_cx, last_cy = data['pos']
                        dist = math.hypot(cx - last_cx, cy - last_cy)
                        # 距离限制稍大一点，防止视频动作太大丢失
                        if dist < 80 and dist < min_dist:
                            min_dist = dist
                            best_match_id = track_id

                    # 更新后台计时器
                    if best_match_id is not None:
                        frames = self.hidden_tracks[best_match_id]['frames'] + 1
                        new_hidden_tracks[best_match_id] = {'frames': frames, 'pos': (cx, cy)}
                    else:
                        new_hidden_tracks[self.next_internal_id] = {'frames': 1, 'pos': (cx, cy)}
                        best_match_id = self.next_internal_id
                        self.next_internal_id += 1

                    current_frames = new_hidden_tracks[best_match_id]['frames']

                    # --- 核心“剧情”逻辑 ---
                    # 如果持续出现的时间 > 设定的阈值，判定为溺水
                    if current_frames > self.drowning_threshold_frames:
                        is_drowning = True
                        self.global_alarm = True
                    else:
                        is_drowning = False

                    # 记录用于右侧显示（不带ID）
                    person_statuses.append(is_drowning)

                    # 准备画面显示元素
                    if is_drowning:
                        box_color = (0, 0, 255)  # 红色
                        text_str = "⚠️ 溺水 (Danger)"
                        texts_to_draw.append((text_str, (x1, max(0, y1 - 25)), (255, 0, 0)))
                    else:
                        box_color = (0, 255, 0)  # 绿色
                        text_str = "✅ 游泳 (Swim)"
                        texts_to_draw.append((text_str, (x1, max(0, y1 - 25)), (0, 255, 0)))

                    # 在 OpenCV 画面中只画框，不画轨迹，不画ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                self.hidden_tracks = new_hidden_tracks

                # --- 更新 UI 和 警报 ---
                if self.global_alarm:
                    self.status_label.config(text="🚨 警报：发现溺水！", bg="red")
                    # 降低警报声音频率（每 20 帧触发一次声音，防止声音重叠）
                    if self.alarm_sound_counter % 20 == 0:
                        self.play_alarm_sound()
                    self.alarm_sound_counter += 1
                else:
                    self.status_label.config(text="🟢 状态正常", bg="green")
                    self.alarm_sound_counter = 0

                # 更新右侧文本框日志 (不再显示难看的ID)
                log_content = f"检测到画面中有 {len(person_statuses)} 人\n"
                log_content += "-" * 22 + "\n"
                for i, p_drown in enumerate(person_statuses):
                    state_str = "❌ 疑似溺水 (报警中)" if p_drown else "✅ 正常游泳"
                    log_content += f"目标目标 : {state_str}\n"

                self.log_text.config(state=tk.NORMAL)
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(tk.END, log_content)
                self.log_text.config(state=tk.DISABLED)

                # --- 转换并使用 PIL 绘制纯净无ID的中文 ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_img)
                font = self.get_chinese_font()

                for text_str, pos, color in texts_to_draw:
                    # 阴影 + 文字
                    draw.text((pos[0] + 1, pos[1] + 1), text_str, font=font, fill=(0, 0, 0))
                    draw.text(pos, text_str, font=font, fill=color)

                self.photo = ImageTk.PhotoImage(image=pil_img)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                self.window.after(10, self.update_video_frame)
            else:
                self.playing_video = False
                self.status_label.config(text="⏹️ 演示完毕", bg="gray")

    def close_app(self):
        self.playing_video = False
        if self.vid is not None:
            self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MockDrowningDemo(root, "防溺水预警系统 ")