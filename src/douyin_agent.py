#!/usr/bin/env python3.11
# douyin_agent.py  智谱 GLM-4 版
import os, subprocess, time, re, html
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

# ---------- tools for agent ----------
def run(cmd: str) -> None :
    subprocess.run(f'adb shell {cmd}', shell=True, check=False)

def dump_hex(s: str) -> str :
    """把字符串每个字符转成 4 位十六进制，空格分隔"""
    return ' '.join(f'{ord(c):04X}' for c in s)

def click(x : float, y : float):
    """点击屏幕上坐标为x,y的地方"""
    w_pix, h_pix = 1080, 2400          # mobilephone屏幕的分辨率，可修改
    tap_x = int((x) * w_pix)
    tap_y = int((y) * h_pix)
    subprocess.run(f'adb shell input tap {tap_x} {tap_y}',shell=True)

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
    # 只留目标类别
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
        return []
    targets = sorted(targets, key=lambda x: x[1], reverse=True)   # 置信度降序
    if index >= len(targets):
        return []
    return targets[index][0].tolist()

@tool
def unlock_mobile(pwd : str = "253918") -> None:
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
def watch_and_choose(label: str, index: int = 0) -> None:
    """上滑动浏览抖音视频，选择其中之一"""
    while((res := check(label,index)) == []):
        run('input swipe 500 1200 500 700 1000')
    click(res[0],res[1])    

@tool
def tap_element(label: str, index: int = 0) -> str:
    """检测并点击抖音屏幕上由label指定类别的元素,index 表示同类别中的下标"""
    xywh = check(label, index)
    if xywh == []:
        return "Not Found!"
    click(xywh[0],xywh[1])
    time.sleep(0.5)
    return f"已点击 {label} 第 {index} 个"

# ==================== 把新工具塞进 Agent ====================
tools = [unlock_mobile,open_douyin,douyin_search]
tools += [tap_element,go_back_to_home]
tools += [watch_and_choose]

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
    order = input("你想让 AI 帮你干什么？\n> ")
    print(agent_executor.invoke({"input": order})["output"])