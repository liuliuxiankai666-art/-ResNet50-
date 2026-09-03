# -ResNet50-
平面图像的粗糙度检测主要集中在图像纹理的识别，纹理的粗细和疏密程度以及GLCM信息都是判断粗糙度等级的重要依据。本库将开源检测代码以及所用数据集。


（1）第一点为内窥图像的环形展开的主要算法部分，为坐标的逆映射并使用最近邻像素算法，具体可查看项目Circular unfold中的完整代码。


（2）最终模型有三个文件：

dataload为数据加载，加载图像与GLCM并进行预处理，并保存提取到的GLCM参数信息。

resnet50为融合改进模型包含各大模块和主要模型。

train为训练代码，修改路径和按需求修改超参数即可进行训练，冻结部分是分为两阶段如果只训练一个数据集就只开启第一部分训练就行，第二阶段是为迁移学习任务而操作的。



（4）最终模型的结果图如下，展示训练集与验证集的准确率/损失函数/mAP等指标，以及混淆矩阵。其他所有指标可从source_complet_report中查询。
<img width="1185" height="584" alt="源数据集训练_train_val_valid_acc_compare" src="https://github.com/user-attachments/assets/704708e9-5a92-4635-bd40-ee8dd056b681" /> <img width="850" height="583" alt="源数据集训练_val_loss" src="https://github.com/user-attachments/assets/2cc50b40-0f4e-4ce1-8f90-65935199530c" />

<img width="850" height="583" alt="源数据集训练_val_map" src="https://github.com/user-attachments/assets/8a463ac1-65bc-4f47-816f-685eb41499d5" /> <img width="1349" height="1349" alt="源数据集训练_best_val_confusion_matrix_6class" src="https://github.com/user-attachments/assets/bf36905b-ac44-4da1-97c8-79f2c531aebf" />


（5）关于独立测试样本部分，同样来源于上面所述的零件，但与图像数据集所用的不一样，现公开部分展示图以及关于独立测试实验的混淆矩阵（上传文件中可直接下载）等，模型名称和矩阵命名一样，分为两种：一是100张总图的测试结果：二是100张中提取的子图测试结果（后缀出现-2）。

<img width="1609" height="708" alt="image" src="https://github.com/user-attachments/assets/68fbe077-67ce-4a2d-85b5-e60f7c872422" />

图中可看到，采集的独立测试样本有外表面和内表面的，以及不同视角。该测试样本未经过任何增强处理以及预处理，主要考验模型对其的泛化能力和预测稳定性。
