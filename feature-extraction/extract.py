#coding:utf-8
import tempfile
import multiprocessing
import os
from staticAnalyzer import run  # 假设你有一个名为staticAnalyzer的模块，并且在其中定义了run函数
from tqdm import tqdm
import shutil
from mamadroid import get_mamadroid_feature
from itertools import repeat

def process_file(file_name):
    tempdir = tempfile.mkdtemp()
    run(file_name, tempdir)
    try:
        shutil.rmtree(tempdir)
    except Exception as e:
        print(e)
    #os.system("rm -rf temp/unpack")

def is_zip_file(file_path):
    """检查文件是否为有效的ZIP文件"""
    try:
        # ZIP文件的魔术数字是前4个字节: 50 4B 03 04 (十六进制)
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'\x50\x4b\x03\x04'
    except Exception as e:
        print(f"检查文件类型时出错: {e}")
        return False

base_dir = "/mnt/hdd/jianwen/downloaded_added/"
def drebin():
    hashlist = os.listdir("features/drebin/")
    hashlist = list(map(lambda x:x.split("drebin-")[0],hashlist))
    #print(hashlist)
    processes = []  # 存储进程对象的列表
    filenames = []
    # 创建进程池
    pool = multiprocessing.Pool(5)
    # 启动多个进程，每个进程处理一个文件
    for file_name in tqdm(os.listdir(base_dir)):
        h = file_name.split('.')[0].upper()
        if h in hashlist:# or not is_zip_file(base_dir+file_name):
            print(h,"is already in hashlist or is not an apk file")
            continue
        filenames.append(base_dir+file_name)
    print(len(filenames)," apks to be processed")
    tqdm(pool.map(process_file, filenames))
    pool.close()
    pool.join()

def mamadroid():
    from pathlib import Path
    hashlist = os.listdir("features/mamadroid/")
    hashlist = list(map(lambda x:x.split(".")[0],hashlist))
    print(hashlist)
    processes = []  # 存储进程对象的列表
    filenames = []
    # 创建进程池
    pool = multiprocessing.Pool(20)
    # 启动多个进程，每个进程处理一个文件
    for file_name in tqdm(os.listdir(base_dir)):
        h = file_name.split('.')[0].upper()
        if h in hashlist:
            print(h,"is already in hashlist")
            continue
        filenames.append(base_dir+file_name)
        #break
    print(len(filenames)," apks to be processed")
    output_names = map(lambda x:"features/mamadroid/"+Path(x).stem+".npz", filenames)
    graph_names = map(lambda x:"features/mamadroid/"+Path(x).stem+".graph", filenames)
    params = list(zip(filenames,output_names,repeat(None)))#graph_names))#
    #print(len(params),params[:10])
    tqdm(pool.starmap(get_mamadroid_feature, params))

    pool.close()
    pool.join()

if __name__ == '__main__':
    drebin()
    #mamadroid()
