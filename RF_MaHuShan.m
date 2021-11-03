%% 最优叶子节点数与树数确定
%%用以训练一个随机森林模型
% 数据以列向量的形式输入
% 示例：
% Input=unifrnd(1,100,[5,140])';
% Output=unifrnd(5,10,[1,140])';
% 实际数据输入时，将自变量命名为Input，数据形式为数值矩阵；将因变量命名为Output


%%此代码有两处需要实时修改的地方
%%1.参数nTree和nLeaf需要根据第一张“Leaef and Trees”图来选取最佳设置值
%%2.精度衡量部分if语句中均方根误差RFRMSE阈值需要依情况设置
%%
clear;
%% 进行数据处理
load('Input.mat');
load('Output.mat');
New_Data_WithoutNaN=deleteNAN([Output,Input]);
Output=New_Data_WithoutNaN(:,1);
Input=New_Data_WithoutNaN(:,2:end);
% 多次循环防止最优结果受到随机干扰
% for RFOptimizationNum=1:5

RFLeaf=[5,10,20,50,100,200,500];  %初始叶子节点数（叶子可看作是每棵树选择特征的个数）
col='rgbcmyk';
figure('Name','RF Leaves and Trees');
for i=1:length(RFLeaf)
    %TreeBagger:参数一（建立树的数量）不宜太少，之后再根据图表选取树的个数
    RFModel=TreeBagger(1000,Input,Output,'Method','r','OOBPrediction','On','MinLeafSize',RFLeaf(i));
    plot(oobError(RFModel),col(i));  %计算不同叶子树时建立的随机森林的袋外误差，并绘制图表
    hold on
end
xlabel('Number of Grown Trees');
ylabel('Mean Squared Error') ;
LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
title(LeafTreelgd,'Number of Leaves');
hold off;

%disp(RFOptimizationNum);
%end


%% 
%% 参数设置
RFRMSEMatrix=[];  %均方根误差矩阵
RFrAllMatrix=[];  %回归系数矩阵

%这里设置一个断点

%观察之前的图表中哪条曲线误差最小，取其值为nTree
%观察该曲线收趋于平缓时最小 的森林树的个数，取其值为nLeaf
nTree=200; %最优树个数
nLeaf=5; %最优叶子节点个数

%% 随机森林建模
% 这些变量用于储存性能最优的森林模型的数据
Best_RFModel=[];
Best_RFPredictConfidenceInterval=[];
Best_RFPredictY=[];
Best_RFr=0;
Best_RFRMSE=10000;
Best_TestX=[];
Best_TestY=[];
Best_TrainX=[];
Best_TrainY=[];
% RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRunNumSet=2000;
for RFCycleRun=1:RFRunNumSet
    %% 训练和测试数据集的划分
    % 划分过程中确保随机性
    RandomNumber=(randperm(length(Output),floor(length(Output)*0.2)))';
    TrainY=Output;
    TestY=zeros(length(RandomNumber),1);
    TrainX=Input;
    TestX=zeros(length(RandomNumber),size(TrainX,2));
    for i=1:length(RandomNumber)
        % 以随机生成的RandomNumber矩阵中的数值当作索引
        % 抽取训练数据中对应的样本数据作为测试数据
        % 被抽取的数据赋值0，代表该处数据已被抽取
        m=RandomNumber(i,1);
        TestY(i,1)=TrainY(m,1);
        TestX(i,:)=TrainX(m,:);
        TrainY(m,1)=0;
        TrainX(m,:)=0;
        % RandomNumber矩阵中的值确保抽取数据的随机性
    end
    % 剔除被随机选中的样本数据
    TrainY(all(TrainY==0,2),:)=[];
    TrainX(all(TrainX==0,2),:)=[];

    %% 随机森林模型的训练
    RFModel=TreeBagger(nTree,TrainX,TrainY,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);
    % RFPredictY为预测结果，RFPredictConfidenceInterval为预测结果的置信区间
    [RFPredictY,RFPredictConfidenceInterval]=predict(RFModel,TestX);
    
    %% 精度衡量
    RFRMSE=sqrt(sum((RFPredictY-TestY).^2)/size(TestY,1));
    % 计算两变量或两矩阵相关系数,corrcoef(x,y):x和y必须大小维度相同
    RFrMatrix=corrcoef(RFPredictY,TestY);
    RFr=RFrMatrix(1,2);
    
    % 这两个矩阵暂时不知道保存着有什么用
    RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
    RFrAllMatrix=[RFrAllMatrix,RFr];

    % 比较当前模型与之前存储的模型
    % 当RMSE与r均比之前优时，将其替换
    if  (RFr>Best_RFr) && (RFRMSE<Best_RFRMSE)
        Best_RFModel=RFModel;
        Best_RFPredictConfidenceInterval=RFPredictConfidenceInterval;
        Best_RFPredictY=RFPredictY;
        Best_RFr=RFr;
        Best_RFRMSE=RFRMSE;
        Best_TestX=TestX;
        Best_TestY=TestY;
        Best_TrainX=TrainX;
        Best_TrainY=TrainY;
        
        % 设置均方根误差的阈值
        % 均方差小于阈值直接退出
        if RFRMSE<0.03
            disp(Best_RFRMSE);
            break;
        end
    end
    disp(RFCycleRun);    
 
%    str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
%    waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
% close(RFScheduleBar);

%% 绘图之前先对各变量的重要性进行归一化
NormalizedImportanceX=[];  % 最重要的变量设置为单位一
MaxOOBError=max(Best_RFModel.OOBPermutedPredictorDeltaError);
for i=1:length(Best_RFModel.OOBPermutedPredictorDeltaError)
    NormalizedImportanceX=[NormalizedImportanceX,Best_RFModel.OOBPermutedPredictorDeltaError(i)/MaxOOBError];
end

%% 变量重要性比较排序
% 绘制每一个自变量对因变量的重要程度图
VariableImportanceX={};
XNum=1;
for i=1:size(Input,2)
    VariableImportanceX{1,XNum}=num2str(i);
    % eval(['VariableImportanceX{1,XNum}=''',i,''';']);
    XNum=XNum+1;
end
figure('Name','Variable Importance Contrast');
VariableImportanceX=categorical(VariableImportanceX);
bar(VariableImportanceX,NormalizedImportanceX);
% xtickangle(45);
set(gca, 'XDir','normal')
xlabel('Factor');
ylabel('Importance');
ylim([-0.5,1.2]);

%% 绘制实测预测对比图
figure
N=1:length(TestY);
plot(N,Best_RFPredictY,'LineWidth',2);
hold on;
plot(N,Best_TestY,'LineWidth',2);
xlabel('Sample');
ylabel('Concentration');
title('Comparison chart of forecast results');
LeafTreelgd=legend({'RFPredictY' '    TestY'},'Location','NorthEast');
hold off;


%% 保存模型
RFModelSavePath='D:\MachineLearning\RF';
save(sprintf('RFModel_MHS.mat',RFModelSavePath),'nLeaf','nTree',...
    'RandomNumber','Best_RFModel','Best_RFPredictConfidenceInterval','Best_RFPredictY','Best_RFr','Best_RFRMSE',...
    'Best_TestX','Best_TestY','Best_TrainX','Best_TrainY','NormalizedImportanceX');
