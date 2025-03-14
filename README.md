# [Enhanced YOLOv10 for Real-time Fish Disease Detection in Aquaculture Farms]()


Duanrui Wang1,2,#, Fan Ge1,3,#, Hongjian Lv1,2, Xingyue Zhu1, Dianting Zeng1, Meng Wu1, Jiyang Yu1,3, Weijian Cui1,2, Chi Wu1,*, Yi Shi2,*
1Southern Marine Science and Engineering Guangdong Laboratory (Guangzhou), Guangzhou, Guang-dong, China.  
2College of Engineering, Shantou University, Shantou, China.  
3Department of Ocean Science and engineering, Southern university of science and   technology, Shenzhen, China.  
#These authors contributed equally to this work and should be co-first considered authors.  
E-mail addresses: qi.wu@gmlab.ac.cn(C. Wu), shy_xflx@163.com(Y. Shi)  
It has now been submitted to《The Visual Computer》  

The dataset can be downloaded from the link: https://www.kaggle.com/datasets/moonburntcat/belangeri-croaker-dataset/data  
It contains folders 1-5 and labels.  

get_FPS.py: get FPS of the model  
heatmap.py: get heapmap of the model,you can choose whatever layers of the model  
train_v10.py: set training parameters here and start training  
test.py: test the model  
The improved YAML file we have tried is located at: DMYOLO\ultralytics\cfg\models\v10  
