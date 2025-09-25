#!/usr/bin/env python3.11
# douyin_agent.py  —— 带“LLM 拆解任务”版本
# 新增依赖：pydantic>=2
# ----------------------------------------------------------
import os, subprocess, time, re, html, glob, json, typing as T
import inspect
os.environ["ZHIPUAI_API_KEY"] = "6fc0d9652ef2492ea13fd43b6a7b0ec0.34Qdu7ColwzG2IFm"

tools = []
from langchain_zhipu import ChatZhipuAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ultralytics import YOLO
import numpy as np
import cv2

planner_llm = ChatZhipuAI(model="glm-4", temperature=0.3)
llm = ChatZhipuAI(  model="glm-4",  temperature=0   )
Img = None
MODEL   = YOLO('best.pt')
CLASSES = [line.strip() for line in open('labels.txt', encoding='utf-8') if line.strip()]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ROOT = os.path.join(BASE_DIR, "../wechat_buttons")
TPL_CACHE = {}

class AtomicTask(BaseModel):
    tool: str = Field(description="必须选自 tools 列表")
    param: T.Union[str, dict] = Field(description="工具所需参数，字符串或 dict")

class Plan(BaseModel):
    tasks: T.List[AtomicTask] = Field(description="顺序执行的原子任务队列")

class WechatSendTextInput(BaseModel):
    receiver: str = Field(
        description="微信接收人",
        AliasChoices={"receiver", "recipient"}
    )
    message:  str = Field(description="要发送的文字")

# --------------  规划 Prompt（Few-shot + JSON） --------------
PLANNER_PROMPT = """你负责把用户自然语言指令拆解成“任务”序列。
可使用的工具如下：
{tool_names}
输出必须严格符合以下 JSON 格式：
{{
  "tasks": [
    {{"tool": "tool_name", "param": "参数字符串或 dict"}}
  ]
}}
【例 1】
用户：打开抖音，搜索“人工智能”，把第一个视频转发给微信好友「张三」
输出：
{{
    "tasks": [
        {{"tool": "open_douyin", "param": ""}},
        {{"tool": "douyin_search", "param": "人工智能"}},
        {{"tool": "choose_at_search", "param": ""}},
        {{"tool": "douyin_share_to_wechat", "param": "张三"}}
    ]
}}
【例 2】
用户：锁屏解锁 1234,去抖音首页刷 10 条视频
输出：
{{
    "tasks": [
        {{"tool": "unlock_mobile", "param": "1234"}},
        {{"tool": "open_douyin", "param": ""}},
        {{"tool": "watch_and_choose_at_home", "param": "10"}}
    ]
}}
【例 3】
用户：打开抖音，搜索“炒鸡教程”，上滑十次，点击进入视频
输出：
{{
    "tasks":[
        {{"tool": "open_douyin", "param":""}},
        {{"tool": "douyin_search", "param":"炒鸡教程"}},
        {{"tool": "up_swipe_at_search", "param":"10"}},
        {{"tool": "choose_at_search", "param":""}}
    ]
}}
现在开始！
用户：{input}
"""

# -------------- 顺序执行 --------------
def run_plan(plan: Plan) -> T.List[str]:
    results = []
    for t in plan.tasks:
        tool_obj = next(i for i in tools if i.name == t.tool)
        arg = t.param
        if isinstance(arg, str):
            sig = inspect.signature(tool_obj.func)
            first_param = list(sig.parameters.keys())[0]
            arg = {first_param: arg}
        res = tool_obj.invoke(arg) or "success"
        results.append(f"{t.tool} -> {res}")
    return results

# --------------  主入口：先规划，再执行 --------------
def execute_order(user_input: str) -> str :
    tool_names = [t.name for t in tools]
    raw = planner_llm.invoke(PLANNER_PROMPT.format(tool_names=tool_names,input=user_input)).content
    raw = re.sub(r'```json|```', '', raw, flags=re.I).strip()
    try:
        plan = Plan.model_validate_json(raw)
    except Exception as e:
        print("[规划失败，退回到 ReAct]", e)
        return agent_executor.invoke({"input": user_input})["output"]

    print("[LLM 规划结果]", plan.model_dump_json(indent=2))
    logs = run_plan(plan)
    return "\n".join(logs)


# ---------- 大模型：智谱 GLM-4 ----------


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
def COME_BACK(tmp : str = "") -> None :
    run('input swipe 0 1200 500 1200 300')
def UP_SWIPE(tmp : str = "") -> None :
    run('input swipe 500 1200 500 700 300')
def SLEEP(x : float) -> None :
    time.sleep(x)

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

def get_screen_size():
    try:
        out = subprocess.check_output(['adb', 'shell', 'wm', 'size'], stderr=subprocess.STDOUT, text=True)
        m = re.search(r'(\d+)[x\s](\d+)', out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 1080, 2340

def click(x : float, y : float, p : bool = True):
    w_pix, h_pix = get_screen_size() 
    tap_x = int((x) * w_pix)
    tap_y = int((y) * h_pix)
    subprocess.run(f'adb shell input tap {tap_x} {tap_y}',shell=True)
    if not p:
        subprocess.run(f'adb shell input swipe {tap_x} {tap_y} {tap_x} {tap_y} 1000',shell=True)

def screenshot() :
    raw = subprocess.check_output('adb shell screencap -p', shell=True)
    raw = raw.replace(b'\r\n', b'\n')          # Windows 换行符
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img

def _wait_tap(label: str, mode: bool = True, max_try: int = 20) -> bool:
    for _ in range(max_try):
        if tap_element(label, mode):
            return True
        SLEEP(0.3)
    return False

def _wait_check(name : str, max_try : int = 20) -> None :
    for _ in range(max_try) :
        SLEEP(0.5)
        xywh = check(name)
        if xywh == [] :
            COME_BACK()
        else:
            break

def At_douyin_home(tmp: str = "") -> None:
    _wait_check('home')

def At_wechat_home(tmp: str = "") -> None:
    _wait_check('wechat_search')

def tap_element(label: str, mode : bool = True, index: int = 0) -> bool:
    xywh = check(label, index)
    if xywh == []:
        print(label,'Not Found!')
        return False
    else :
        print(label,'Found!')
    click(xywh[0],xywh[1],mode)
    SLEEP(0.5)
    return True

def check(label: str, index: int = 0) -> list[float]:
    global Img
    Img = screenshot()
    results = MODEL(Img, conf=0.35)[0]
    if results.boxes is None:
        return []
    targets = []
    for b in results.boxes:
        cls_name = CLASSES[int(b.cls[0])]
        if cls_name == label:
            targets.append((b.xywhn[0], b.conf[0]))
    if not targets:
        boxes = match_template(label, Img)
        h_pix, w_pix = Img.shape[:2]
        for (x1,y1,x2,y2) in boxes:
            xc = (x1+x2)/2/w_pix
            yc = (y1+y2)/2/h_pix
            ww = (x2-x1)/w_pix
            hh = (y2-y1)/h_pix
            targets.append(([xc, yc, ww, hh], 1.0))
    if not targets:
        print(label,'Not Found!')
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
        UP_SWIPE()
        SLEEP(0.5)
        if pwd:
            run(f'input text {pwd}')
            run('input keyevent 66')

@tool
def open_douyin(tmp: str) -> None:
    """打开抖音"""
    run('monkey -p com.ss.android.ugc.aweme 1')
    SLEEP(2)

@tool           
def douyin_search(keyword: str) -> None:
    """在抖音主页搜索keyword""" 
    At_douyin_home()
    if not _wait_tap('search'):
        return
    SLEEP(2)
    run('input tap 545 155')
    run(f'am broadcast -a ADB_INPUT_TEXT --es msg {keyword}')
    SLEEP(0.5)
    run('input tap 990 150')
    SLEEP(2)

@tool 
def up_swipe_at_search(times: str = '0') -> None:
    """在搜索区上滑，次数=times"""
    for _ in range(int(times)) :
        UP_SWIPE()

@tool
def choose_at_search(tmp : str) -> None:
    """点击检测到的第一个视频"""
    while((res := check('video',0)) == []):
        UP_SWIPE()
    click(res[0],res[1])
    SLEEP(1)
    _wait_tap('ads',True,2)

@tool
def watch_and_choose_at_home(times: str) -> None:
    """在抖音首页上滑指定次数浏览视频"""
    At_douyin_home()
    for _ in range(int(times)):
        UP_SWIPE()
        SLEEP(0.5)

@tool
def open_wechat(tmp: str = "") -> None:
    """打开微信"""
    run('monkey -p com.tencent.mm 1')
    SLEEP(2)

def wechat_search_friend(name : str) -> None :
    At_wechat_home()
    if not _wait_tap('wechat_search') :
        return
    if not _wait_tap('search_frame') :
        return
    run(f'am broadcast -a ADB_INPUT_TEXT --es msg {name}')
    SLEEP(1)
    run('input tap 570 460')    

@tool(args_schema=WechatSendTextInput)
def wechat_send_message(receiver: str, message : str) -> None:
    """在微信首页搜索receiver,打开聊天窗口,给receiver发送message"""
    wechat_search_friend(receiver)
    if not _wait_tap('wechat_input') :
        return
    run(f'am broadcast -a ADB_INPUT_TEXT --es msg {message}')
    if not _wait_tap('wechat_send') :
        return
    
@tool
def douyin_share_to_wechat(name: str) -> None:
    """分享抖音上某个视频链接给微信上的name"""
    if not _wait_tap('share') :
        return
    SLEEP(1)
    while((res := check('sharelink',0))==[]):
        run('input swipe 560 2120 320 2120 1000')
    click(res[0],res[1])
    if not _wait_tap('wechat') :
        return
    SLEEP(2)
    wechat_search_friend(name)
    SLEEP(1)
    if not _wait_tap('wechat_input',False) :
        return
    if not _wait_tap('wechat_paste') :
        return
    if not _wait_tap('wechat_send') :
        return

# ==================== 把新工具塞进 Agent ====================
tools = [douyin_share_to_wechat,wechat_send_message,unlock_mobile,up_swipe_at_search]
tools += [open_wechat,open_douyin,choose_at_search,watch_and_choose_at_home,douyin_search]

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
    print(execute_order(order))