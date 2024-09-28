# Rovas实验框架

## 一、实验设置

### 1.1 实验数据集（已统一格式）

#### 1、真实异常检测数据集（主要来自ADBench，以及其他基准中使用的数据集）

pendigits、optdigits、Waveform、shuttle、PageBlocks、satellite、credit-card-fraud、WPBC

#### 2、真实多分类数据集（部分异常检测数据集，以及对应的原始多分类数据集，采集自UCI/Kaggle/OpenML）

adult、apple、balita、drybean、Iris、Obesity、Wine

#### 3、合成异常检测数据集（包含Local/Global/Cluster等不同异常类型和异常比例）

合成自apple、Iris、Optidigits、Waveform、Wine(将原始真实数据集的异常全部删除，然后加入合成的异常)

#### 4、真实异常检测数据集（对应采集自LMU的真实的含不同比例异常的数据集）

Annthyroid、Cardiotocography、PageBlocks、Wilt



### 1.2 基准算法（已实现）

#### 1、传统无监督异常检测算法（包含基于统计/基于距离/基于密度/基于集成/基于角度/基于主动学习等）

ECOD、COPOD、OCSVM、LOF、COF、SOD、LODA、IForest、ABOD

#### 2、无监督深度异常检测算法（主流）

DeepSVDD、RCA、GOAD、ICL（较新）、DIF（较新）、SLAD（较新）、DAGMM（实验的过程需要借助标签，疑似有问题）

#### 3、弱监督深度异常检测算法（高准确度）

DeepSAD、DevNet、PReNet（较新，效果很好）、RoSAS（较新，效果很好）、XGBOD

#### 4、监督深度异常检测算法（可能缺乏监督训练数据）

LGB（效果很好）、CatB（效果很好）、XGBoost（效果很好）



### 1.3 实验指标（已实现）

Accuracy(适用于类别分布均匀的情形)

F1-score/Precision/Recall(适用于类别分布不均)

ROC-AUC(适用于二分类)

PRAUC(适用于类别分布不均，用于平衡precision和recall)

AP(适用于类别分布不均，精确度和召回率的加权平均)

CD图(比较多算法模型性能)

其他指标(时间，内存占用，鲁棒性等，需要时添加)



### 1.4 分类器选取（已实现）

多分类SVM，softmax(待替换)，RandomForest(待替换)，NaiveBayes，KNN，DecisionTree



## 二、实验设计（按重要性从高到低排序，1为最重要）

### 1、Rovas修复效果的实验

实验点1：借助Rovas修复ugly outliers后，对下游分类器（如SVM分类器）的性能（如Accuracy）影响

...

### 2、Rovas与基准算法对比

实验点1：对比Rovas和基准在传统异常检测领域（在正常值和异常值两种类型的分类数据中检测异常值的能力）的性能（如异常检测器的Accuracy）差异，关于检测的传统异常类型可以参照ADBench中划分的local/global/Dependency/Cluster。后期可以考虑添加基准算法对good/bad/ugly的检测性能（关于Rovas对这三类检测能力在后面单独讨论）

...

### 3、Rovas对ugly outliers的检测能力（建议与1一起做）

实验点1：首先是Rovas能发现多少比例的ugly outliers，其次是在不同异常类型（local/global/Dependency/Cluster）和异常比例（合成数据集/真实数据集的不同异常比例）下，Rovas能发现的ugly outliers比例如何变化。



### 4、Rovas不同修复方法对比（建议与1,3一起做）

实验点1：基于统计/直接删除/基于ML模型等不同修复方法下，修复ugly outliers的不同效果（体现在下游分类器的Accuracy等指标）



### 5、Rovas鲁棒性测试

对比在不同异常比例下（合成数据集/真实数据集的不同异常比例），以及不同异常类型下（分别改变异常类型为local/global/Dependency/Cluster，以及分别改变为good/bad/ugly），Rovas修复性能（如Accuracy）差异



### 6、Rovas可扩展性

随数据集的扩展（行，列，不同领域数据集，真实/合成数据集），资源消耗的变化（时间/内存占用等），在可扩展性测试中可重点关注两大类：1）Rovas部分谓词的可扩展性； 2）Rovas几种规则形式的可扩展性。