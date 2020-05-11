
import os
import shutil
import pandas as pd
import random
from PIL import Image
import json

def convertjpg(jpgfile,width=224,height=224):
    try:
        img=Image.open(jpgfile)
        new_img = img.resize((width, height), Image.BILINEAR)
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        new_img.save(jpgfile)
    except Exception as e:
        print(e,"删除")
        try:
            os.remove(jpgfile)
        except Exception:
            shutil.rmtree(jpgfile)

rootdir = os.path.join(os.path.split(os.path.realpath(__file__))[0] + "/bird") +"/"
birds = os.listdir(rootdir)
birdli = []

for bird in birds:
    print(f"doing {bird}")
    work_dir = rootdir + bird
    content = os.listdir(work_dir)
    for i,filename in enumerate(content):
        convertjpg(os.path.join(work_dir, filename))
        newfile = str(i) + ".jpg"
        # print(newfile)
        # 实现重命名操作
        try:
            os.rename(
                os.path.join(work_dir , filename),
                os.path.join(work_dir , newfile)
            )
        except Exception:
            try:
                os.rename(
                    os.path.join(work_dir, filename),
                    os.path.join(work_dir,str(i)+"n" + newfile)
                )
            except Exception:
                pass
        # except Exception:
            # pass
#         pat = "bird/" + bird + "/" + i + " " + bird
#         birdli.append(pat)
dic = {}
for i,bird in enumerate(birds):
    work_dir = rootdir + bird
    os.rename(work_dir,rootdir + "n" + str(i))
    dic[str(i)] = bird

birds = os.listdir(rootdir)
for i,bird in enumerate(birds):
    work_dir = rootdir + bird
    bird = bird.replace("n","")
    os.rename(work_dir,rootdir + bird)

with open(rootdir + "../dict.json","w") as f:   
    f.truncate()
    
    f.write(json.dumps(dic))


birds = os.listdir(rootdir)
for bird in birds:
    # print(f"doing {bird}")
    content = os.listdir(rootdir + bird)
    for i in content:
#         batch_rename(rootdir + bird, "jpg")
        pat = "bird/" + bird.strip() + "/" + i.replace(" ","") + " " + bird.strip()
        birdli.append(pat)
#     num = len(content) - 1
#     if num >= 20:
#         sum += 1
#         print(bird)
#         birdli.append(bird)
#         print("===="*32)

# birdli
# birds
random.shuffle(birdli)
num = len(birdli)
print(f"总数据：{num}")
pay = num//10

trnum = pay * 8
vanum = pay * 1
train = birdli[0:trnum]
vail = birdli[trnum:trnum + vanum]
test = birdli[trnum + vanum:-1]
with open(rootdir + "../train_extra_list.txt","w") as f:
    f.truncate()
    for i in train:
        f.write(i + "\n")
with open(rootdir + "../train_list.txt","w") as f:
    f.truncate()
    for i in test:
        f.write(i + "\n")
with open(rootdir + "../val_list.txt","w") as f:   
    f.truncate()
    for i in test:
        f.write(i + "\n")

