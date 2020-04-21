<font size=6>**Uncertainty Experiments**</font> 
@[toc]
# Introduction for This Uncertainty Experiments
此Repository实现了一种Uncertainty的计算，是主要基于论文What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision？的复现。分别对回归任务和分割任务进行了实验。

# Experiments:
## 1、Uncertainty on Regression Task
详见文件`uncertainty_sin.py`
在玩具数据（带有噪声的sin曲线）上进行拟合，并实现Epistemic Uncertainty 和 Aleatoric Uncertainty 的计算。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200421210043651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RhaXRlZA==,size_16,color_FFFFFF,t_70)

## 2、Uncertainty on Segmentation Task
详见文件夹`Bag_Data_Segmentation_Experiments`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200421210108951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RhaXRlZA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200421210123577.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RhaXRlZA==,size_16,color_FFFFFF,t_70)
