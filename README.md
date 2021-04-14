# 基于HeteroFL论文的个人学习实践


## Quick steps to run the experiment
 - Step1:基于anaconda配置该实验的虚拟环境（可选）。
 - Step2:pip install -r requirements.txt 
 - Step3:根据自己的设备CPU/GPU。 例如：设置torch.cuda.manual_seed() (GPU)/torch.manual_seed() (CPU)
 - Step4:config.yml可用于设置你的各种参数，包含实验的默认参数。  
 - Step5:run train_transformer_fed.py 开始你的实验吧！

## Tips

 - data.py用于获取数据集，目前提供了MNIST和CIFAR10数据集。
 - fed.py里实现了联邦评估，联邦训练（全局模型训练参数的分发，和本地模型训练参数的汇总等功能。
 - local_process.py中包含了对本地模型的操作 从联邦训练中获取最新参数和训练本地模型等。
 - 训练结果通过tensorboard实现了可视化，结果存在./src/output文件下。
 - utils.py的process_control()方法中存放了实验的超参数。

## 个人的论文分析（待补充）
飞书文档链接: https://ku3ejouhjy.feishu.cn/docs/doccnywLr5rZUNLty5FwRgB9aLe   


