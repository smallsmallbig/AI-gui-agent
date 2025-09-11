import uiautomator2 as u2
import time

d = u2.connect()  # 需手机开启USB调试
d.app_start("com.tencent.mm")
time.sleep(3)
d.screenshot("data/screenshots/wechat_home.png")
print("微信已打开并截图保存")