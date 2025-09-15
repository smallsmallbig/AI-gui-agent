from pathlib import Path

shots  = Path('shots_train')
manual = Path('labels/manual')

for img in shots.glob('*.jpg'):
    txt = manual / f'{img.stem}.txt'
    if not txt.exists():
        txt.touch()      # 创建 0 字节空文件

print('空标签已补齐，可无标注训练')