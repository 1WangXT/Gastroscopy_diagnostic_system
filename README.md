## Gastroscopy_diagnostic_system胃镜病灶辅助诊断系统

- 本项目设计一个符合临床使用的、集胃镜图像解剖部位分类、病理分类、胃早癌病灶分割、胃早癌病灶检测一体的智能辅助诊断系统。

《2020年中国恶性肿瘤报告》指出，我国胃癌发病率和死亡率均居恶性肿瘤的<u><span>第三位</span></u>

我国是个消化道疾病高发的国家，其中胃癌的5年存活率高达百分之95，而中晚期的五年存活率则不足百分之5，所以早发现早诊断是治疗胃癌的有效途径。在人工筛查中，往往无法快速定位图像解剖结构，医生自身的问题造成误诊漏诊，胃镜图像多，但有效图像少，所导致医生筛查的工作量大。因此，我们要做的系统要能够帮助医生大规模筛查胃部疾病，能够智能辨识图像解剖结构部位及病变区域，能够减轻医生负担，降低漏诊率，能够提高诊断准确率和效率的系统。

🎯1、软件启动界面

软件启动界面如图所示，该界面的功能有：

<img src="file:///C:/Users/%E6%89%B6%E8%8B%8F/Pictures/Typedown/fd1876de-8e24-4aeb-a49c-416c7c95c216.png" title="" alt="fd1876de-8e24-4aeb-a49c-416c7c95c216" data-align="center">

（1）在软件启动界面中显示软件的代表性标志等信息；

（2）提前加载库（python工具包、数据库链接、数据库信息加载等），启动界面时载入运行软件所需要的文件，避免用户在一种盲目状态下等待，缓解用户等待的焦躁。



🎯2、用户登录界面

用户登录界面如图所示，该界面的功能有：

<img src="file:///C:/Users/%E6%89%B6%E8%8B%8F/Pictures/Typedown/2cfdf0a0-7494-41fa-9c09-f868c3c02b74.png" title="" alt="2cfdf0a0-7494-41fa-9c09-f868c3c02b74" data-align="center">

（1）用户名密码验证：与数据库相连，如果数据库中存在该用户的信息，则登录成功，否则失败；

（2）关联注册界面：点击用户注册，跳转到注册界面；

（3）关联数据库管理界面：点击数据库进入数据库界面，管理用户信息；

（4）其他功能。



🎯3、用户注册界面

用户注册界面如图所示，该界面的功能有：

<img src="file:///C:/Users/%E6%89%B6%E8%8B%8F/Pictures/Typedown/9be5c726-a4f0-456f-8078-61b31d79b121.png" title="" alt="9be5c726-a4f0-456f-8078-61b31d79b121" data-align="center">

（1）用户注册：与数据库相连，注册时，如果用户名不存在，且密码无误，即可注册成功。

（2）注册成功之后，该用户则会被后台添加到数据库中。回到登录界面，即可登录成功。



🎯4、管理员数据库管理界面

在程序中的配置文件中进行本地数据库的用户名、密码、ip地址、端口信息、数据库名字的配置，即可绑定该用户数据库。管理员数据库管理界面如图所示，该界面的功能有：

<img title="" src="file:///C:/Users/%E6%89%B6%E8%8B%8F/Pictures/Typedown/7b6cb302-ff57-40dd-bd5b-b8217a0ba3c9.png" alt="7b6cb302-ff57-40dd-bd5b-b8217a0ba3c9" data-align="center">

（1）管理员身份验证：与数据库相连，如果数据库中存在该管理员的信息，则登录成功。

（2）管理员登录之后，连接本地数据库，展示数据库中的用户列表信息。

（3）添加、删除、修改用户：管理员登录之后，可根据用户id对该用户的信息进行增删改。



🎯5、胃镜病灶辅助诊断界面

胃镜病灶辅助诊断界面是本项目的主界面，也是功能实现的主界面。如图所示，该界面的功能有：

<img src="file:///C:/Users/%E6%89%B6%E8%8B%8F/Pictures/Typedown/8ae7ddd7-7447-4469-bda2-a5bfb7c8704d.png" title="" alt="8ae7ddd7-7447-4469-bda2-a5bfb7c8704d" data-align="center">

（1）功能选择模块

可选择四个功能，也是本项目的重点功能，包括：

胃部解剖部位视频/图像分类模块：这个模块主要用于对胃镜的胃部初步检查的视频中的胃部解剖部位进行实时检测并生成相对应的胃部图像或按照类别分成文件夹，以帮助相关医务人员进行初步筛查，一定程度上减少医务人员因经验不足等原因造成检查时的胃部部位区分错误的现象。

胃部病理图像分类模块：这个模块主要对医务人员截取下来的重点图像进行分类处理，通过这个模块可以输出对重点图像处理后的分类结果，将不同病理，如正常病人或胃早癌病人区分开，为医务人员判断患者是否患有胃早癌提供一定的参考。

胃早癌病灶图像分割模块：这个模块主要对医务人员筛选出的胃早癌图像进行处理，通过这个模块可以输出对胃早癌图像处理后的分割结果，直观的将胃早癌区域的形状、大小、边界等信息较为精细的标记出来，为医务人员确定胃早癌病灶位置形状提供一定的参考。

胃早癌病灶检测模块：这个模块主要对医务人员筛选出的胃早癌图像进行检测，通过这个模块可以直观的将胃早癌的区域用矩形框标记出来，并且标记该区域患有胃早癌的概率，为医务人员判断胃早癌病灶位置和概率提供一定的参考。

（2）支持图像或视频单个输入或批量输入：对于单个处理的图像或视频，通过数据链加载到中央计算机中预先训练好的算法中之后，得到的输出的结果会直观展现在视图上；而批量处理一般和自动保存结合使用，选择文件夹进行输入即可根据所选功能批量处理文件夹中的所有图像或视频，保存在相应文件夹。

（3）模型选择模块

根据四种不同的功能，共包含了15种不同的网络模型，可以选择不同模型对输入图像或视频进行处理。分类模型有：ConvNeXt-MA（推荐）、ConvNeXt、MobileNet、ResNet、Swin-Transformer；病灶分割模型有：U2-CFF（推荐）、U2-Net、U-Net、DeepLabV3、LR-ASPP；病灶检测模型有：YOLOv5-RC（推荐）、YOLOv5、YOLOv3、Faster-RCNN、RetinaNet。用户或医护人员可以根据速度和准确率选择自己需要的模型。

（4）输入文件按钮：支持图像、视频作为输入。当选择单个处理图像或视频时，打开的是某个文件；而选择批量处理时，打开的则是某个文件夹。

（5）选择是否自动保存检测到的结果：在选择批量处理时基本会和该功能一起使用。当处理分类目标时，会对图像所处的文件夹中的所有数据按照其不同类别分成不同的文件夹保存到本地；当处理胃早癌分割和病灶检测时，则会将处理好的结果保存到一个本地文件夹中。

（6）结果统计功能：可以直观为用户展示出目前所处理的图像所在的解剖部位信息或胃早癌病灶区域的数量，在批量处理文件时该功能不生效。

（7）播放/暂停/停止按钮：控制输入在模型中处理的流程和视图中的展示。



## Environmental installation

```bash
# CONDA creates Python virtual environment
conda create -n learn_pyqt5 python=3.8
# Activate environment
conda activate learn_pyqt5

# Installation Library
pip install -r requirements.txt

# Modify the MySQL server configuration information in `/config/config.py` in file

# Add a database to the database and run `/model/db.sql` file

# Pack
# It is packaged into many files. It is recommended to use it when it is very dependent
pyinstaller pyqt5_example.spec

# Package into a separate exe. It is recommended to use small files
# One drawback is that it will first read into memory and decompress the dependency to the cache directory.
# If the application is large, it is recommended to package it into a folder
pyinstaller pyqt5_example_exe.spec
```
