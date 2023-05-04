import os
import time

script_name = "pipeline_twiter.py"
num_runs = 5


for i in range(num_runs):
    print(f"开始执行第{i+1}次实验...")
    os.system(f"python {script_name}")
    print(f"第{i+1}次实验执行完成！")


    if i < num_runs - 1:
        print("等待5秒后重新运行实验...")
        time.sleep(5)
