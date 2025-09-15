# open_douyin.py
# 手机用USB数据线连接，打开开发者模式
# 打开手机，打开抖音

import subprocess, time

def run(cmd):
    subprocess.run(f'adb shell {cmd}', check=True)

def screen_on() -> bool:
    out = subprocess.check_output('adb shell dumpsys power', shell=True, text=True)
    return 'mHoldingDisplaySuspendBlocker=true' in out

def unlocked() -> bool:
    out = subprocess.check_output('adb shell dumpsys window', shell=True, text=True)
    keys = ['mStatusBarKeyguard=false',
            'isKeyguardShowing=false',
            'mShowingLockscreen=false']
    return any(k in out for k in keys)

if not screen_on():
    run('input keyevent KEYCODE_WAKEUP')

if not unlocked():
    run('input swipe 500 1200 500 400 300')
    time.sleep(0.5)
    pwd = input('密码：').strip()
    if pwd:
        run(f'input text {pwd}')
        run('input keyevent 66')

print('--- 打开抖音')
run('input keyevent KEYCODE_HOME')
time.sleep(0.5)
run('monkey -p com.ss.android.ugc.aweme 1')
print('抖音已打开！')

time.sleep(2) #等待抖音界面加载完全
run('input tap 990 150')   # 搜索的坐标

keyword = input('想搜什么视频：')
run(f'am broadcast -a ADB_INPUT_TEXT --es msg "{keyword}"')

run('input tap 990 150')    # 点击搜索
print('已进入搜索结果，上下滑即可看视频！')

