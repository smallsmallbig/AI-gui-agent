#!/usr/bin/env python3.11
# douyin_agent.py  智谱 GLM-4 版
import os, subprocess, time, re, html
import glob
os.environ["ZHIPUAI_API_KEY"] = "6fc0d9652ef2492ea13fd43b6a7b0ec0.34Qdu7ColwzG2IFm"
from langchain_zhipu import ChatZhipuAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain import hub
from ultralytics import YOLO
import numpy as np
import cv2

# ---------- 大模型：智谱 GLM-4 ----------
llm = ChatZhipuAI(  model="glm-4",  temperature=0   )
Img = None
MODEL   = YOLO('best.pt')
CLASSES = [line.strip() for line in open('labels.txt', encoding='utf-8') if line.strip()]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ROOT = os.path.join(BASE_DIR, "../wechat_buttons")
TPL_CACHE = {}

# ---------- Agent ----------
from langchain.prompts import PromptTemplate
template = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: {input}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}"""

# ---------- tools ----------
def _load_templates():
    for f in glob.glob(f"{TEMPLATE_ROOT}/*.png"):
        name = os.path.basename(f).rsplit(".",1)[0]   # 文件名做 label
        tpl  = cv2.imread(f, cv2.IMREAD_COLOR)
        TPL_CACHE[name] = tpl

def match_template(label: str, img: np.ndarray, thresh: float = 0.85):
    """返回 [(xmin, ymin, xmax, ymax), ...] 列表，可能多个"""
    if label not in TPL_CACHE:
        return []
    tpl = TPL_CACHE[label]
    h, w = tpl.shape[:2]
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= thresh)
    boxes = []
    for y, x in zip(*loc):
        boxes.append((x, y, x+w, y+h))
    # 简单 NMS，防止重叠
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    for b in boxes:
        if all([max(abs(b[0]-k[0]), abs(b[1]-k[1]))>10 for k in keep]):
            keep.append(b)
    return keep
def run(cmd: str) -> None :
    subprocess.run(f'adb shell {cmd}', shell=True, check=False)

def dump_hex(s: str) -> str :
    """把字符串每个字符转成 4 位十六进制，空格分隔"""
    return ' '.join(f'{ord(c):04X}' for c in s)

def click(x : float, y : float, p : bool):
    """点击屏幕上坐标为x,y的地方"""
    w_pix, h_pix = 1080, 2340          # mobilephone屏幕的分辨率，可修改
    tap_x = int((x) * w_pix)
    tap_y = int((y) * h_pix)
    subprocess.run(f'adb shell input tap {tap_x} {tap_y}',shell=True)
    if p == False:
        subprocess.run(f'adb shell input swipe {tap_x} {tap_y} {tap_x} {tap_y} 1000',shell=True)

def screenshot() :
    """返回当前屏幕截图"""
    raw = subprocess.check_output('adb shell screencap -p', shell=True)
    raw = raw.replace(b'\r\n', b'\n')          # Windows 换行符
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img

def check(label: str, index: int = 0) -> list[float]:
    global Img
    Img = screenshot()
    results = MODEL(Img, conf=0.35)[0]
    if results.boxes is None:
        return [0,0,0,0]
    label = html.unescape(label)
    label = label.replace("label=","")
    label = label.replace("_button","")
    label_raw = re.search(r'[a-zA-Z0-9_]+', label).group(0)
    label_clean = label_raw.lower()
    targets = []
    for b in results.boxes:
        cls_name_raw = html.unescape(CLASSES[int(b.cls[0])])
        cls_name_raw = cls_name_raw.replace("_btn","")
        cls_name_clean = re.sub(r'[^a-zA-Z0-9_]', '', cls_name_raw).lower()
        if cls_name_clean == label_clean:
            targets.append((b.xywhn[0], b.conf[0]))
    if not targets:
        boxes = match_template(label_clean, Img)
        h_pix, w_pix = Img.shape[:2]
        for (x1,y1,x2,y2) in boxes:
            xc = (x1+x2)/2/w_pix
            yc = (y1+y2)/2/h_pix
            ww = (x2-x1)/w_pix
            hh = (y2-y1)/h_pix
            targets.append(([xc, yc, ww, hh], 1.0))
    if not targets:
        print(label_clean)
        print('Targets empty')
        return []
    targets = sorted(targets, key=lambda x: x[1], reverse=True)   # 置信度降序
    if index >= len(targets):
        return []
    box, _conf = targets[index]
    return box.tolist() if hasattr(box, 'tolist') else box

@tool
def unlock_mobile(pwd : str = "") -> None:
    """解锁手机"""
    def screen_on() -> bool:
        out = subprocess.check_output('adb shell dumpsys power', shell=True).decode('utf-8', errors='ignore')
        return 'mHoldingDisplaySuspendBlocker=true' in out
    def unlocked() -> bool:
        win = subprocess.check_output(['adb', 'shell', 'dumpsys', 'window'],encoding='utf-8', errors='ignore')
        for token in ('isKeyguardShowing=false','mKeyguardShown=false','mKeyguardShowing=false'):
            if token in win:
                return True
        return not any(k in win for k in ('Keyguard', 'keyguard'))
    if not screen_on():
        run('input keyevent KEYCODE_WAKEUP')
    if not unlocked():
        run('input swipe 500 1200 500 400 300')
        time.sleep(0.5)
        if pwd:
            run(f'input text {pwd}')
            run('input keyevent 66')

@tool
def open_douyin(tmp: str) -> None:
    """打开抖音"""
    run('monkey -p com.ss.android.ugc.aweme 1')
    time.sleep(2)

@tool 
def go_back_to_home(tmp: str) -> None:
    """返回抖音首页"""
    for _ in range(20) :
        xywh = check('home',0)
        if xywh == [] :
            run('input swipe 0 1200 500 1200 300')
            time.sleep(0.5)
        else:
            break

@tool            
def douyin_search(keyword: str) -> None:
    """搜索keyword""" 
    tap_element('search',0)
    time.sleep(2)
    run('input tap 545 155')          # 点搜索框
    keyword = re.sub(r"^keyword=", "", keyword) 
    run(f'am broadcast -a ADB_INPUT_TEXT --es msg {keyword}')
    time.sleep(0.5)
    run('input tap 990 150')          # 点搜索
    time.sleep(2)

@tool
def watch_and_choose(label: str = 'video', index: int = 0) -> None:
    """上滑浏览抖音视频，选择其中之一"""
    while((res := check(label,index)) == []):
        run('input swipe 500 1200 500 700 1000')
    click(res[0],res[1],True)

@tool 
def douyin_share_to_wechat(name: str) -> None:
    """分享抖音上某个视频链接给微信上的name"""
    tap_element('share',0)
    time.sleep(0.5)
    while((res := check('sharelink',0))==[]):
        run('input swipe 560 2120 320 2120 1000')
    click(res[0],res[1],True)
    tres = False
    for _ in range(20):
        tres = tap_element('wechat_1',0)
        if tres == True :
            break
    if tres == False :
        return
    for _ in range(20):
        tres = tap_element('wechat_search',0)
        if tres == True :
            break
    if tres == False :
        return
    for _ in range(20):
        tres = tap_element('search_frame',0)
        if tres == True :
            break
    if tres == False :
        return
    name = re.sub(r"^name=", "", name) 
    run(f'am broadcast -a ADB_INPUT_TEXT --es msg {name}')
    time.sleep(2)
    run('input tap 570 460')
    time.sleep(0.5) 
    for _ in range(20) :
        res = check('input',0)
        if res != [] :
            break
    if res != [] :
        click(res[0],res[1],False)
    else :
        return
    for _ in range(20):
        tres = tap_element('paste',0)
        if tres == True :
            break
    if tres == False :
        return
    for _ in range(20):
        tres = tap_element('send',0)
        if tres == True :
            break
    if tres == False :
        return

@tool
def tap_element(label: str, index: int = 0) -> bool:
    """检测并点击抖音屏幕上由label指定类别的元素,index 表示同类别中的下标"""
    xywh = check(label, index)
    if xywh == []:
        print(label,'Not Found!')
        return False
    else :
        print(label,'Found!')
    click(xywh[0],xywh[1],True)
    time.sleep(0.5)
    return True

# ==================== 把新工具塞进 Agent ====================
tools = [unlock_mobile,open_douyin,douyin_search]
tools += [tap_element,go_back_to_home]
tools += [watch_and_choose,douyin_share_to_wechat]

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=20,
    handle_parsing_errors=True
)
# ---------- main ----------
if __name__ == "__main__":
    _load_templates()
    print("已加载的模板:", list(TPL_CACHE.keys()))
    order = input("你想让 AI 帮你干什么？\n> ")
    print(agent_executor.invoke({"input": order})["output"])