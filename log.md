## exp 0

linear regression，country和status都用one hot编码，且没有drop year列

MSE is 5994.2995251433495
R2 score is 0.6144327801073088

## exp 1

linear regression，country和status都用one hot编码，且drop了year列

MSE is 5994.746152903936
R2 score is 0.614404051975981


## exp 3

gbm，使用one hot 编码

MSE is 6214.088293231035
R2 score is 0.6277494233839982

## exp 4

gbm，但是抛弃country和status

MSE is 6126.966916720464
R2 score is 0.6329683680000517

## exp 5

尝试 drop 一些 column，最后决定每个都试试

结果：

----------------------------------------
Dropped Year
MSE is 6085.540632124799
R2 score is 0.6354499803621729
----------------------------------------
Dropped Life expectancy 
MSE is 6725.443797577873
R2 score is 0.5971170325381476
----------------------------------------
Dropped infant deaths
MSE is 6206.427496617813
R2 score is 0.6282083380022079
----------------------------------------
Dropped Alcohol
MSE is 6234.702863537092
R2 score is 0.6265145220885853
----------------------------------------
Dropped percentage expenditure
MSE is 6167.703812840289
R2 score is 0.6305280529683729
----------------------------------------
Dropped Hepatitis B
MSE is 6212.915219372132
R2 score is 0.6278196955461857
----------------------------------------
Dropped Measles 
MSE is 6249.356712087246
R2 score is 0.6256366936260608
----------------------------------------
Dropped  BMI 
MSE is 6279.783190784433
R2 score is 0.6238140168784903
----------------------------------------
Dropped under-five deaths 
MSE is 6203.358655526598
R2 score is 0.6283921747634253
----------------------------------------
Dropped Polio
MSE is 6187.78977299995
R2 score is 0.6293248176909687
----------------------------------------
Dropped Total expenditure
MSE is 6350.0218168825195
R2 score is 0.6196064215190535
----------------------------------------
Dropped Diphtheria 
MSE is 6215.003400134191
R2 score is 0.6276946045503591
----------------------------------------
Dropped  HIV/AIDS
MSE is 6468.064729482055
R2 score is 0.6125351440915225
----------------------------------------
Dropped GDP
MSE is 6183.262511101677
R2 score is 0.6295960201220574
----------------------------------------
Dropped Population
MSE is 6193.609822766783
R2 score is 0.6289761717143743
----------------------------------------
Dropped  thinness  1-19 years
MSE is 6189.199914384355
R2 score is 0.6292403441658618
----------------------------------------
Dropped  thinness 5-9 years
MSE is 6185.889960936577
R2 score is 0.6294386245927508
----------------------------------------
Dropped Income composition of resources
MSE is 6144.814143518668
R2 score is 0.6318992424657613
----------------------------------------
Dropped Schooling
MSE is 6271.2534109928665
R2 score is 0.6243249873211338

其中 'Year', 'Income composition of resources' 都可以直接丢掉，丢掉后的结果：

MSE is 6007.033829164127
R2 score is 0.6401528750252923

然而如此一来，在测试集上的分数居然还降低了，遂放弃

目前最高分数是将 status 离散化，放弃country，放弃year，为55.85

## exp 6

即 exp_5_bagging

在测试集上的成绩为58.48，在本地为0.631

参数：
```python
regressor = BaggingRegressor(
    base_estimator=None,
    n_estimators=480,
    max_samples=1.0,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    warm_start=False,
        random_state=42
)
```

## exp 7

即 exp_6_bagging

新的更高分：58.59

参数：
```python
regressor = BaggingRegressor(
    base_estimator=None,
    n_estimators=1280,
    max_samples=0.5,
    max_features=0.6,
    bootstrap=False,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    random_state=42
)
```

看起来可以先无脑堆 n_estimators 的数量。

失败了，堆到1920后降低为58.33

## exp 8

即 exp_7_bagging

突发奇想，如果Status只有两类，那岂不是只需要一个onehot向量就够了？完全不需要两列啊

然而删去一列之后分数居然降低了

## exp 9

即 exp_8_bagging

突然想到，这样参数条件下的bagging其实就不是bagging了，而是 pasting 和 随机子空间的结合，那如果我把 bootstrap 和 oob_score 全都改成 True 会怎么样呢？

结果是57.43... 更差了
