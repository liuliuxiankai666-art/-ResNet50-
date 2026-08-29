# -ResNet50-
平面图像的粗糙度检测主要集中在图像纹理的识别，纹理的粗细和疏密程度以及GLCM信息都是判断粗糙度等级的重要依据。本库将开源检测代码以及所用数据集。


（1）第一点为内窥图像的环形展开的主要算法部分，为坐标的逆映射并使用最近邻像素算法，具体可查看项目Circular unfold中的完整代码。


（2）最终模型有三个文件：

dataload为数据加载，加载图像与GLCM并进行预处理，并保存提取到的GLCM参数信息。

resnet50为融合改进模型包含各大模块和主要模型。

train为训练代码，修改路径和按需求修改超参数即可进行训练，冻结部分是分为两阶段如果只训练一个数据集就只开启第一部分训练就行，第二阶段是为迁移学习任务而操作的。

（3）关于样块和工件的来源：

主要待检测工件是本文联系常州市金坛城西大陆机械配件有限公司按照Ra0.1、Ra0.2、Ra0.4和Ra0.8目标等级加工的，出厂时要求加工人员用粗糙度测量仪做了检测达到合格标准后我才验收的，可以附带当时工作人员返回给我的一张图，没有记录的习惯所以只拍了这个图给我。

<img width="810" height="1440" alt="ddacac24890fd0a4af1a8fadda83bbd5" src="https://github.com/user-attachments/assets/557cc606-e23a-4ffd-b275-549d0012d2c0" />

按照国家标准（GB/T6062-2009）表面粗糙度Ra0.1等级的真值范围为0.08-0.12微米即可。

样块是本人从 淘宝 西量旗舰店 购买的前后陆续购买了十几次用于实验，同时也从淘宝各大商店购买了几十个小轴承套（Ra0.4-1.6都有）用于丰富样本和检测。现在做轴承套厂商普遍都是做Ra0.1-0.8的轴承套，很少能用上Ra0.05和很粗糙度的Ra1.6及以上，所以本项目是专注于利用Ra0.05-1.6去有容差检测轴承套Ra0.1-0.8。

<img width="3072" height="4096" alt="63debf1d2cfea4de9932e0fa9c2a5c65" src="https://github.com/user-attachments/assets/6f4a4463-db08-4e87-ab95-ba929b52df3c" />


采集平台的硬件与环境条件话可以查看我的论文。
（4）最终模型的结果图如下，展示训练集与验证集的准确率/损失函数/mAP等指标，以及混淆矩阵。其他所有指标可从source_complet_report中查询。
<img width="2370" height="1166" alt="源数据集训练_train_val_valid_acc_compare" src="https://github.com/user-attachments/assets/704708e9-5a92-4635-bd40-ee8dd056b681" />
<img width="1770" height="1166" alt="源数据集训练_val_loss" src="https://github.com/user-attachments/assets/2cc50b40-0f4e-4ce1-8f90-65935199530c" />
<img width="1770" height="1166" alt="源数据集训练_val_map" src="https://github.com/user-attachments/assets/8a463ac1-65bc-4f47-816f-685eb41499d5" />
<img width="2699" height="2366" alt="源数据集训练_best_val_confusion_matrix_6class" src="https://github.com/user-attachments/assets/bf36905b-ac44-4da1-97c8-79f2c531aebf" />

（5）
