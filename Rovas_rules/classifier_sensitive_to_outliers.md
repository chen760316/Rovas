## 一、Kimi认定的对异常值敏感的机器学习分类器

机器学习中，一些分类器对异常值比较敏感，主要是指那些在训练过程中对个别数据点赋予较高权重，从而使得模型对这些点特别敏感的算法。以下是一些对异常值比较敏感的分类器：

> 1、决策树：由于决策树是基于特征选择和数据划分的，异常值可能会对树的生成产生较大影响，特别是当异常值在特征空间中较为孤立时；
> 2、支持向量机(SVM)：SVM在定义决策边界时依赖于支持向量，如果异常值接近决策边界，它们可能会成为支持向量，从而影响模型的性能 ；
>
> 3、随机森林：尽管随机森林通过集成多个决策树来提高模型的稳定性，但如果异常值在多个树中都被选择为分裂节点，它们仍然可能对整体模型产生影响 ；
>
> 4、梯度提升树：与随机森林类似，梯度提升树也是基于树的集成方法，异常值如果在多次迭代中被用作分裂节点，也可能影响最终模型 ；
>
> 5、K-近邻(KNN)：KNN算法根据邻近的K个样本进行预测，如果异常值距离其他样本较远，它们可能会对预测结果产生较大影响 ；
>
> 6、AdaBoost：AdaBoost通过迭代地调整数据权重来增强模型对困难样本的学习，异常值如果在迭代中获得较高的权重，可能会对最终的强分类器产生较大影响 ；
>
> 7、基于深度学习的模型：深度学习模型，如自编码器，可能会在重构过程中对异常值特别敏感，因为它们尝试学习数据的分布，异常值可能会扭曲这个分布。

## 二、ChatGPT认定的对异常值敏感的机器学习分类器

在机器学习中，一些分类器对异常值（也称为离群点或异常数据）比较敏感。以下是一些常见的分类器和它们对异常值的敏感程度：

**线性回归和逻辑回归**：这类模型基于最小二乘法或对数几率回归，通常对异常值比较敏感，因为异常值会影响模型的参数估计，尤其是在回归模型中，异常值可以显著影响回归线的位置。

**支持向量机（SVM）**：标准的SVM模型对于异常值比较敏感，尤其是在没有使用正则化参数时。异常值可以影响支持向量的选择，从而影响分类决策边界。使用SVM时可以调整`C`参数来处理异常值的影响，但在面对极端异常值时，SVM仍可能受影响。

**k-最近邻（k-NN）**：k-NN分类器基于邻近数据进行分类，对异常值较为敏感。异常值可能会影响最近邻的计算，从而影响分类结果。特别是在异常值距离很近的情况下，它们可能会干扰正常数据点的分类。

**决策树**：虽然决策树对异常值不如线性模型或k-NN敏感，但在某些情况下，异常值仍然可能影响树的结构和分裂过程，尤其是在树深度较大时。异常值可能导致决策树过拟合训练数据。

**随机森林**：随机森林通常对异常值较为鲁棒，因为它通过多个决策树进行集成，通常可以减少单棵树对异常值的敏感性。然而，如果异常值非常显著，仍可能对模型产生影响。

**朴素贝叶斯**：朴素贝叶斯分类器通常对异常值的敏感性较低，因为它基于特征的条件独立性假设，并且通常不直接受到异常值的影响。然而，如果异常值导致特征分布的显著改变，可能会对模型产生一定影响。

## 三、Kimi修复关系数据中的异常值

**删除异常值**：如果**异常值数量不多，且不影响数据分析的总体结构**，可以选择直接删除含有异常值的记录。这是一种简单直接的处理方式，但可能会减少数据量并影响数据的完整性 。

**替换异常值**：可以用一些统计度量来替换异常值，例如**均值、中位数或众数**。例如，使用数据列的均值来替换异常值是一种常见的做法 。

**使用模型预测**：如果**数据集有足够的特征能够描述异常值**，可以使用机器学习模型来预测并替换这些异常值 。可以使用其他模型（如回归模型）预测异常值的合理值。这个方法适用于数据的异常值可以通过其他特征的模式来预测的情况。

**基于近邻的方法**：**使用K最近邻（KNN）算法，根据距离找到与异常值最近的K个点，并用这些点来估计并替换异常值** 。

**变换数据**：对数据进行变换，如**对数变换或Box-Cox变换**，以减少异常值的影响 。

**设置阈值**：对于某些连续变量，**可以设置一个最大值和最小值的阈值，将超过这个范围的值进行封顶或底部处理** 

**使用箱型图（Boxplot）**：箱型图可以帮助识别数据中的异常值。**异常值通常被定义为小于第一四分位数减去1.5倍四分位距或大于第三四分位数加上1.5倍四分位距的值** 。

**稳健的估计方法**：**使用稳健的统计方法，如使用中位数绝对偏差（MAD）来确定异常值** 

**组合方法**：在某些情况下，可以组合使用以上方法，以达到更好的修复效果 

## 四、异常值的常用检测方法

#### 1、简单统计方法

> - 直接观察整体数据，使用pandas.describe()观察数据的统计描述(统计数据应当为连续性)
> - 简单使用散点图，从图中即可看出

#### 1、GRUBBS TEST

> Grubbs’ test 是一个假设检验方法。
>  原假设 H0：数据集中无异常值；备择假设H1：数据集中有一个异常值。

#### 2、3σ原则（Z分数法）

> **数据分布需要服从正态分布**或者近似正态分布

#### 2、稳健Z分数法（中位数绝对偏差法）

> 也被称为中位数绝对偏差法。它类似于Z-score方法，只是参数有所变化。由于平均值和标准差受异常值的影响很大，因此我们使用中位数和中位数的绝对偏差来改变这个参数。

#### 3、箱型图（四分位距法）

> 使用箱型图的**四方位距(IQR)**对异常进行检测，也叫\**Tukey's test**
>
> 一般使用IQR的1.5倍为标准:值大于**上四分位+1.5\*IQR**为异常值,值小于"**下四分位-1.5\*IQR**"为异常值

#### 3、截尾处理

> 方法类似于IQR法。如果一个值超过了第99个百分位数的值，并且低于给定值的第1个百分位数，则被视为异常值。

#### 4、基于模型检测

> 这种方法一般会构建一个**概率分布模型**，并计算对象符合该模型的概率，把具有低概率的对象视为异常点
>
> - 如果模型是簇的集合，则异常是不显著属于任何簇的对象；
>
> - 如果模型是回归时，异常是相对远离预测值的对象。
>
>   离群点的概率定义:离群点是一个对象，关于数据的概率分布模型，它具有低概率。这种情况的前提必须知道数据集服从什么分布，如果估计错误就造成了重尾分布。
>
>   优点：具有坚实的统计基础，当**存在充分的数据和所用的检验类型的知识**时，这些检验可能非常有效
>
>   缺点：对于多元数据，可用的选择少一些，并且**对于高维数据，这些检测可能性很差**

#### 5、基于近邻度的离群点检测

> 统计方法是利用数据的分布来观察异常值，一些方法甚至需要一些分布条件，而**在实际中数据的分布很难达到一些假设条件**。
>
> **确定数据集的有意义的邻近性度量比确定它的统计分布更容易**。这种方法比统计学方法更一般、更容易使用，因为一个对象的离群点得分由到它的k-最近邻（KNN）的距离给定。
>
> 离群点得分对 k的取值高度敏感。**如果k太小，则少量的邻近离群点可能导致较低的离群点得分；如果 K太大，则点数少于 k的簇中所有的对象可能都成了离群点**。为了使该方案对于 k的选取更具有鲁棒性，可以使用 k个最近邻的平均距离。
>
> 优点：操作简单
>
> 缺点：基于邻近度的方法需要 O(m2)时间，大数据集不适用；对参数的选择也敏感；不能处理具有不同密度区域的数据集，因为它使用全局阈值，不能考虑这种密度的变化
>

#### 6、基于密度的异常检测

> 从基于密度的观点来说，离群点是在低密度区的对象。基于密度的离群点检测与基于邻近度的离群点检测密切相关，因为**密度通常用邻进度定义**。
>
> 一种常用的定义密度的方法是，**定义密度为到k个最近邻的平均距离的倒数**。如果该距离小，则密度高，反之亦然。另一种密度定义是使用DBSCAN聚类算法使用的密度定义，即**一个对象周围的密度等于该对象指定距离 d 内对象的个数**。
>
> 优点：给出了对象是离群点的定量度量，并且即使数据具有不同的区域也能够很好的处理
>
> 缺点：这些方法必然具有 O(m2)的时间复杂度；参数选择是困难的。
>

#### 7、基于聚类的异常检测

> **一个对象是基于聚类的离群点，如果该对象不属于任何簇，那么该对象属于离群点**。
>
> **DBScan** 是一种用于把数据聚成组的聚类算法。它同样也被用于单维或多维数据的基于密度的异常检测。其它聚类算法比如 **k 均值和层次聚类**也可用于检测离群点。
>
> 优点：线性和接近线性复杂度（k均值）的聚类技术来发现离群点可能是**高度有效的**；簇的定义通常是离群点的补，**因此可能同时发现簇和离群点**；
>
> 缺点：产生的离群点集和它们的得分可能**非常依赖所用的簇的个数和数据中离群点的存在性**；聚类算法产生的**簇的质量对该算法产生的离群点的质量影响非常大**

#### 8、专用的离群值检测

> 两个专门用于检测异常点的方法比较常用：One Class SVM和Isolation Forest
>
> 孤立森林：孤立森林是一种**无监督学习算法**，属于**组合决策树家族**。它**明确地隔离异常值**, 而不是通过给每个数据点分配一个分数来分析和构造正常的点和区域。它利用了这样一个事实：**异常值只是少数，并且它们具有与正常实例非常不同的属性值。**该算法**适用于高维数据集**，并且被证明是一种非常有效的异常检测方法。
>
> **Random Cut Forest (RCF)** ：该算法是亚马逊用于异常检测的无监督算法。它也通过关联异常分数来工作。较低的分数值表示数据点点是正常的，较高的分数值则表示数据中存在异常。**「低」和「高」**的定义取决于应用，但一般实践表明**，超过平均分三个标准差的分数**被认为是异常的。这个算法的优点在于它可以**处理非常高维的数据**。它还可以**处理实时流数据**（内置 AWS Kinesis Analytics）和**离线数据**。

#### 9、可视化数据

> 数据可视化对于数据清理、数据挖掘、异常值和异常组的检测、趋势和集群识别等都很有用。下面是用于发现异常值的数据可视化图列表。
>
> 1. Box and whisker plot (box plot). 箱线图
> 2. Scatter plot. 散点图
> 3. Histogram. 直方图
> 4. Distribution Plot. 分布图
> 5. QQ plot. Q-Q图

## 五、异常值常用的处理方法

总体来说：

- 如果是**类别变量，可以用众数来代表总体**。
- 如果是**连续变量，可以用平均数或中位数来代表总体**。如果数据的**异常值、极端异常值比较多，那么尽可能使用中位数**；如果**没有太明显的异常值，平均数更具有代表性**。

#### 1、删除含有异常值的记录

> 如果由于**数据输入错误、数据处理错误或异常值观测值非常小，我们会删除异常值**。我们还可以在两端使用修剪来去除异常值。但是**当数据集很小的时候，删除观测结果并不是一个好主意**。

2、视为缺失值：将异常值视为缺失值，按照缺失值进行处理

3、平均值修正：可用前后两个观测值的平均值修正该异常值

4、不处理：不直接在具有异常值的数据集上进行数据挖掘

#### 5、转换值

> 转换变量也可以消除异常值。这些转换后的值减少了由极值引起的变化。
>
> 1. 范围缩放
> 2. 对数变换
> 3. 立方根归一化
> 4. Box-Cox转换
>
> 这些技术将数据集中的值转换为更小的值。**如果数据有很多极端值或倾斜，此方法有助于使您的数据正常**。但是这些技巧并不总是给你最好的结果。从这些方法中不会丢失数据。在所有这些方法中，**box-cox变换给出了最好的结果**。

#### 6、插补法

> 像缺失值的归责一样，我们也可以归责异常值。在这种方法中，我们可以使用平均值、中位数、零值。由于我们进行了输入，所以没有丢失数据。这里的中值是合适的，因为它不受异常值的影响。

#### 7、分组处理

> 在统计模型中，**如果离群值较多且数据集较小，则应将它们分开处理**。其中一种方法是将两个组视为两个不同的组，为两个组建立单独的模型，然后结合输出。**但是，当数据集很大时，这种技术是繁琐的**。