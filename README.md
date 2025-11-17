# [Real-time Fish Disease Detection System for Aquaculture Based on Edge Computing]()

<img width="853" height="471" alt="image" src="https://github.com/user-attachments/assets/dab12010-97ff-4b23-8745-ac4fa1a90250" />

Duanrui Wang, Fan Ge, Hongjian Lv, Xingyue Zhu, Dianting Zeng, Meng Wu, Jiyang Yu, Weijian Cui, Qiwei Qin, Shaowen Wang4, Chi Wu, Yi Shi   
E-mail addresses: qi.wu@gmlab.ac.cn(C. Wu), shy_xflx@163.com(Y. Shi)  
It has now been submitted to《Aquaculture》  
<img width="843" height="378" alt="image" src="https://github.com/user-attachments/assets/65b47ff8-b91c-4fdd-bde1-53c944f47a29" />

Dataset  
A custom dataset named the Belangeri Croaker Dataset was constructed by collecting images of both infected and healthy belangeri croaker. The images were captured from the Penghu semi-submersible deep-sea cage situated in proximity to Guishan Island, China. The monitoring equipment terminal was also deployed at this location. On March 19, 2024, the underwater camera was installed at a depth of 4 meters on the cage railing, and continuous image collection of belangeri croaker was conducted over a three-week period. The fish had been cultured for five months at the time of data collection.
The dataset can be downloaded from the link: https://www.kaggle.com/datasets/moonburntcat/belangeri-croaker-dataset/data  
It contains folders 1-5 and labels.  
<img width="565" height="368" alt="image" src="https://github.com/user-attachments/assets/882a05ba-9a3a-4168-8299-a66e9acdaf39" />

How to Use the Source Code  
requirements.txt: the enviroment requirements
get_FPS.py: get FPS of the model  
heatmap.py: get heapmap of the model,you can choose whatever layers of the model  
train_v10.py: set training parameters here and start training  
test.py: test the model  
The improved YAML file we have tried is located at: DMYOLO\ultralytics\cfg\models\v10  

Contributions & Contact Us  
Email us directly at [15652582873@163.com].
