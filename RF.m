%% ����Ҷ�ӽڵ���������ȷ��
%%����ѵ��һ�����ɭ��ģ��
% ����������������ʽ����
% ʾ����
% Input=unifrnd(1,100,[5,140])';
% Output=unifrnd(5,10,[1,140])';
% ʵ����������ʱ�����Ա�������ΪInput��������ʽΪ��ֵ���󣻽����������ΪOutput


%%�˴�����������Ҫʵʱ�޸ĵĵط�
%%1.����nTree��nLeaf��Ҫ���ݵ�һ�š�Leaef and Trees��ͼ��ѡȡ�������ֵ
%%2.���Ⱥ�������if����о��������RFRMSE��ֵ��Ҫ���������
%%
clear;

load('Input.mat');
load('Output.mat');
Input=[X1,X2,X3,X4];
Output=Y;
% ���ѭ����ֹ���Ž���ܵ��������
% for RFOptimizationNum=1:5

RFLeaf=[5,10,20,50,100,200,500];  %��ʼҶ�ӽڵ�����Ҷ�ӿɿ�����ÿ����ѡ�������ĸ�����
col='rgbcmyk';
figure('Name','RF Leaves and Trees');
for i=1:length(RFLeaf)
    %TreeBagger:����һ��������������������̫�٣�֮���ٸ���ͼ��ѡȡ���ĸ���
    RFModel=TreeBagger(1000,Input,Output,'Method','r','OOBPrediction','On','MinLeafSize',RFLeaf(i));
    plot(oobError(RFModel),col(i));  %���㲻ͬҶ����ʱ���������ɭ�ֵĴ�����������ͼ��
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
%% ��������
RFRMSEMatrix=[];  %������������
RFrAllMatrix=[];  %�ع�ϵ������

%��������һ���ϵ�

%�۲�֮ǰ��ͼ�����������������С��ȡ��ֵΪnTree
%�۲������������ƽ��ʱ��С ��ɭ�����ĸ�����ȡ��ֵΪnLeaf
nTree=100; %����������
nLeaf=5; %����Ҷ�ӽڵ����

%% ���ɭ�ֽ�ģ
% RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRunNumSet=2000;
for RFCycleRun=1:RFRunNumSet
    %% ѵ���Ͳ������ݼ��Ļ���
    % ���ֹ�����ȷ�������
    RandomNumber=(randperm(length(Output),floor(length(Output)*0.2)))';
    TrainY=Output;
    TestY=zeros(length(RandomNumber),1);
    TrainX=Input;
    TestX=zeros(length(RandomNumber),size(TrainX,2));
    for i=1:length(RandomNumber)
        % ��������ɵ�RandomNumber�����е���ֵ��������
        % ��ȡѵ�������ж�Ӧ������������Ϊ��������
        % ����ȡ�����ݸ�ֵ0������ô������ѱ���ȡ
        m=RandomNumber(i,1);
        TestY(i,1)=TrainY(m,1);
        TestX(i,:)=TrainX(m,:);
        TrainY(m,1)=0;
        TrainX(m,:)=0;
        % RandomNumber�����е�ֵȷ����ȡ���ݵ������
    end
    % �޳������ѡ�е���������
    TrainY(all(TrainY==0,2),:)=[];
    TrainX(all(TrainX==0,2),:)=[];

    %% ���ɭ��ģ�͵�ѵ��
    RFModel=TreeBagger(nTree,TrainX,TrainY,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);
    % RFPredictYΪԤ������RFPredictConfidenceIntervalΪԤ��������������
    [RFPredictY,RFPredictConfidenceInterval]=predict(RFModel,TestX);
    
    %% ���Ⱥ���
    RFRMSE=sqrt(sum((RFPredictY-TestY).^2)/size(TestY,1));
    % ���������������������ϵ��,corrcoef(x,y):x��y�����Сά����ͬ
    
    %% ���ж�������ڵ��Դ���
    % if isnan(RFRMSE)
    %     keyboard;
    % end
    %%
    RFrMatrix=corrcoef(RFPredictY,TestY);
    RFr=RFrMatrix(1,2);
    RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
    RFrAllMatrix=[RFrAllMatrix,RFr];
    %���þ�����������ֵ
    if RFRMSE<8000
        disp(RFRMSE);
        break;
    end
    disp(RFCycleRun);
%    str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
%    waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
% close(RFScheduleBar);

%% ��ͼ֮ǰ�ȶԸ���������Ҫ�Խ��й�һ��
NormalizedImportanceX=[];  % ����Ҫ�ı�������Ϊ��λһ
MaxOOBError=max(RFModel.OOBPermutedPredictorDeltaError);
for i=1:length(RFModel.OOBPermutedPredictorDeltaError)
    NormalizedImportanceX=[NormalizedImportanceX,RFModel.OOBPermutedPredictorDeltaError(i)/MaxOOBError];
end

%% ������Ҫ�ԱȽ�����
% ����ÿһ���Ա��������������Ҫ�̶�ͼ
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
ylim([0,1.2]);

%% ����ģ��
RFModelSavePath='D:\MachineLearning\RF';
save(sprintf('RFModel.mat',RFModelSavePath),'nLeaf','nTree',...
    'RandomNumber','RFModel','RFPredictConfidenceInterval','RFPredictY','RFr','RFRMSE',...
    'TestX','TestY','TrainX','TrainY','NormalizedImportanceX');