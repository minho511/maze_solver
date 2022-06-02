import os
import shutil
direction = ['d','f','l', 'r', 'u']
for d in direction:
    sourcePath = f'./imgs/data/{d}'
    targetPath = f'./dataset/{d}'
    for (root, dirs, files) in os.walk(sourcePath):
        for idx, f in enumerate(files):
            # print(idx)
            shutil.copy(os.path.join(sourcePath, f), os.path.join(targetPath, f'maze{idx+1}.png'))