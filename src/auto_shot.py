#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time
import uiautomator2 as u2
from tqdm import tqdm
from pynput import keyboard

# ---------- 参数 ----------
d = u2.connect_usb()
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shots'))
os.makedirs(save_dir, exist_ok=True)
total = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# ---------- 暂停控制 ----------
paused = False
def on_press(key):
    global paused
    if key == keyboard.Key.space:
        paused = not paused
        print('\n[PAUSED]' if paused else '\n[RESUMED]')

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ---------- 主循环 ----------
print('按空格可暂停/继续，Ctrl+C 终止')
for i in tqdm(range(total), desc='Screenshot'):
    while paused:          # 空转等待
        time.sleep(0.1)
    d.swipe(0.5, 0.8, 0.5, 0.2, duration=0.08)
    time.sleep(0.5)
    d.screenshot(os.path.join(save_dir, f'{i:04d}.jpg'))

print(f'✅ 截图完成，共 {total} 张，保存在 {save_dir}')
listener.stop()