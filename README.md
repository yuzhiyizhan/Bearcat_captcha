# 熊猫不定长验证码识别

## 本自述文件会自述以下内容
##### 1.项目环境安装与启动
        1.1 环境安装
	
        1.2 快速运行
##### 2.项目结构描述与细节
        2.1 项目结构描述

        2.2 细节描述
	
##### 3.识别验证码的思路

##### 4.遇到的错误和解决方法

***注意任何时候你都应该备份你的数据集，数据集来之不易***

***注意任何时候你都应该备份你的数据集，数据集来之不易***

***注意任何时候你都应该备份你的数据集，数据集来之不易***

本人使用的环境为:
CPU:lntel(R)Core(TM)i7-7700HQ CPU@2.80GHz

GPU:NVDIA GeForce GTX 1060

IDE:pycharm

使用cpu训练和gpu训练使用的网络结构不一样，后面会提到

项目不会提供任何模型，所有模型需自行训练(只提供渔，不提供鱼)

# 1.项目环境安装与启动
## 1.1 环境安装
tennsorflow2.1无法使用CTC

本项目在tensorflow2.2或2.3下面都可以运行(2.4未发布)

但是两种的安装方法都有区别下面详细说一下(windowns环境):

拉取项目

git clone https://gitclone.com/github.com/yuzhiyizhan/Bearcat_captcha

git clone https://github.com/yuzhiyizhan/Bearcat_captcha

CPU的直接命令行 

pip install tensorflow==2.2 -i https://pypi.douban.com

然后

pip install -r requirements.txt -i https://pypi.douban.com/simple
	
	tennsorflow2.1
    1.安装CUDA 11版本 (官网)[https://developer.nvidia.com/cuda-toolkit]
    2.由于CUDA会自动配好环境本项目不在详述 在命令行输入 nvcc -V 查看CUDA版本
    3.安装conda (推荐在清华镜像站下载Anaconda或者Miniconda都可以)
    4.更新一下conda (conda update -n base conda)
    5.创建python3.7.7的虚拟环境并进入 (conda create -n example python=3.7.7) (conda activate example)
    6.安装tensorflow2.1 (conda install tensorflow-gpu)
    7.再安装其他依赖 (pip install -r requirements.txt -i https://pypi.douban.com/simple)

    tensorflow2.2
    1.安装CUDA 11版本 (官网)[https://developer.nvidia.com/cuda-toolkit]
    2.由于CUDA会自动配好环境本项目不在详述 在命令行输入 nvcc -V 查看CUDA版本
    3.安装conda (推荐在清华镜像站下载Anaconda或者Miniconda都可以)
    4.更新一下conda (conda update -n base conda)
    5.创建python3.7.7的虚拟环境并进入 (conda create -n example python=3.7.7) (conda activate example)
    6.安装tensorflow2.2 (pip install tensorflow-gpu==2.2 -i https://pypi.douban.com/simple)
    7.安装cudnn (conda install cudatoolkit=10.1 cudnn=7.6.5)
    8.再安装其他依赖 (pip install -r requirements.txt -i https://pypi.douban.com/simple)
    
    tensorflow2.3
    1.安装CUDA 11版本 (官网)[https://developer.nvidia.com/cuda-toolkit]
    2.由于CUDA会自动配好环境本项目不在详述 在命令行输入 nvcc -V 查看CUDA版本
    3.安装conda (推荐在清华镜像站下载Anaconda或者Miniconda都可以)
    4.更新一下conda (conda update -n base conda)
    5.创建python3.7.7的虚拟环境并进入 (conda create -n example python=3.7.7) (conda activate example)
    6.安装tensorflow2.3 (pip install tensorflow-gpu==2.3 -i https://pypi.douban.com/simple)
    7.安装cudnn (conda install cudatoolkit=10.1 cudnn=7.6.5)
    8.再安装其他依赖 (pip install -r requirements.txt -i https://pypi.douban.com/simple)
    
## 1.2 快速运行

### 注意(项目基于tensorflow2.2(2.3也可以))

### 以下说明的是中英数识别，图片分类(例如九宫格)

### 运行项目
    ps:不想自己练的拉取分支
    直接运行app.py
    默认开启5006端口,post请求接受一个参数img
    需要base64一下,具体请看spider_example.py
	十分不建议，因为我也不知道自己写了啥
    
### 第一步:新建项目

    运行New_work.py(两个参数第一个是项目路径，第二个是项目名字)

### 第二步:初始化工作路径

    运行init_working_space.py

### 第三步:准备标注好的数据(图片名为，便签_一串可以找到你图片的哈希，MD5什么都可以，但是要唯一)

    1.将训练数据放到train_dataset文件夹

    2.将验证数据放到validation_dataset文件夹

    3.将测试数据放到test_dataset文件夹

### 如果你的标注数据是一坨的话按照下面步骤区分开来(必须先区分好数据在进行下一步)

    1.将一坨数据放到train_dataset文件夹(一坨指的是全部数据集在同一文件夹内)

    2.运行move_path.py

### 如果你暂时没有数据,不用慌,先用生成的数据集吧

    运行gen_sample_by_captcha.py

### 第四步:修改配置文件
	选择要训练的类型
	
	ORDINARY 默认模式(解决不定长，显存小，快速理解)
	
	NUM_CLASSES 图片分类(解决例如12306，九宫格)
	
	CTC识别文字，不需要设置长度(解决不定长，显存占用非常大，大概需要16G)
	
	CTC_TINY识别文字，需要设置长度(解决不定长，显存占用比较大，大概需要6G)
	
	EFFICIENTDET目标检测 (解决点选，显存占用比较大，最少需要6G)
	
	运行cheak_file.py查看自己的数据最大高和宽
	
	IMAGE_HEIGHT和IMAGE_WIDTH最好设置的比数据集的高宽要大
	
	本项目对小于配置文件高宽图片的处理是填充
	
	大于配置文件高宽的图片先进行等比缩小然后再填充

### 第五步:打包数据

	注意:目标检测使用数据生成器这一步跳过
    运行pack_dataset.py


### 第六步:编写模型并编译(model)

    暂时先使用项目自带的模型吧


### 第七步:开始训练

    运行train.py

    
### 第八步:开启可视化(这步可以省略)

    tensorboard --logdir "logs"

### 第九步:评估模型

    丹药出来后要看一下是几品丹药
    运行test.py

### 第十步:开启后端

    运行app.py
    python app.py
    
### 第十一步:调用接口

    先运行本项目给的例子感受一下
    注意：这是微博的验证码(普通英数类型)
    python spider_example.py

## 下面开始补充刚刚省略的一些地方,由于设置文件备注比较完善，解释部分参数

### MODE
    目前一共五种
    'ORDINARY'      默认模式
    'NUM_CLASSES'   图片分类
    'CTC'           文字识别
    'CTC_TINY'	    文字识别
    'EFFICIENTDET'     目标检测
    

### 是否使用数据增强(数据集多的时候不需要用)
    DATA_ENHANCEMENT = False
数据集不够或者过拟合时，可以考虑数据增强下

增强方法在Function_API.py里面的Image_Processing.preprosess_save_images

### 验证码的长度
    CAPTCHA_LENGTH = 8
    
这个数字要取你要识别验证码的最大长度

否则会报错raise ValueError

注意CTC和NUMCLASSES和EFFICIENTDET模式这个参数不再起作用

### BATCH_SIZE

    BATCH_SIZE = 16

如果你的设备顶得住，可以尝试调大点

### 训练次数

    EPOCHS = 200

请放心调有训练多少轮验证损失下不去，停止训练的回调设置

还有断点续训的回调设置

    EARLY_PATIENCE = 8
    
其他设置如果没有特别情况，尽量不要改

# 2.项目结构描述

## 2.1项目结构描述

## 文件夹

### works
    工作目录

### App_model
    后端模型保存路径

### checkpoint
    保存检查点
    
### CSVLogger
    把训练轮结果数据流到 csv 文件
    
### label
	标签存放路径

### logs
    保存被 TensorBoard 分析的日志文件

### model
    保存模型
    
### train_dataset
    保存训练集
    
### train_enhance_dataset
    保存增强后的训练集
    
### train_pack_dataset
    保存打包好的训练集
    
### validation_dataset
    保存验证集
    
### vailidation_pack_dataset
    保存打包好的验证集
    
### test_dataset
    保存测试集
    
    
## 文件

### New_work.py
    新建工作目录

### app.py
    开启后端

### callback.py
    回调函数参考
    [keras中文官网](https://keras.io/zh/callbacks/)
    运行该文件会返回一个损失最小的权重文件
    
### captcha_config.json
    生成验证码的配置文件
      "image_suffix": "jpg",生成验证码的后缀
      "count": 20000,生成验证码的数量
      "char_count": [4, 5, 6],生成验证码的长度
      "width": 100,生成验证码的宽度
      "height": 60，生成验证码的高度
	  
### cheak_file.py
	检查数据集图片的高和宽

### delete_file.py
    删除所有数据集的文件
    这里是防止数据太多手动删不动
    
### utils.py
    项目核心，三大类
    Image_Processing
    图片处理和标签处理
    WriteTFRecord
    打包数据集
    Predict_Image
    预测类模型生成后用这个类来预测和部署
    
### gen_sample_by_captcha.py
    生成验证码
    
### init_working_space.py
    初始化工作目录
    ***注意:此文件只在第一次运行项目时运行***
    ***因为这会重置checkpoint CSVLogger logs***
    
### models.py
    搭建模型网络，运行会生成model.png，展示模型的结构
	需要安装graphviz，官网下载地址为[graphviz](http://www.graphviz.org/)
	
### move_path.py
	区分数据集
	
### num_classes.json
	运行pack_dataset.py后产生
	记录网络输出的数字对应哪个值
	映射表
    
### pack_dataset.py
    打包数据集
  
### save_model.py
    把损失最小的检查点保存成模型

### settings.py
    项目的设置文件
    
### spider_example.py
    爬虫调用例子
    返回return_code状态码
    return_info 处理状态
    result 识别结果
    recognition_rate 每个字符的识别率
	time 识别时间单位s

### test.py
    读取模型进行测试

### train.py
    开始训练

## 2.2 细节描述
### 目标检测(EFFICIENTDET)
	把数据集放到	train_dataset
	把标签放到		label
	[标签格式](https://github.com/yuzhiyizhan/generate_click_captcha)
	运行 move_path区分数据集(可省略)
	运行 train.py开始训练
	
### 标签与本项目描述的不符
	修改utils.py
	
	Image_Processing.extraction_image
	提取全部的图片
	
	Image_Processing.extraction_label
	根据图片名获取标签，将标签保存到列表里
	去重后将之保存到json，然后读取json
	把标签转化为张量
	
	修改pack_dataset.py
	读取全部图片与标签进行打包
	
	以上为数据处理的过程，根据自己的需要自行修改
	
### 部署
	修改app.py
	
	后端用的gradio(简单而且提供一个前端界面)
	
	以上为模型的过程，根据自己的需要自行修改
	
### 使用CPU或GPU训练
	运行比较大的模型爆显存十分正常
	这时就需要大佬们给我们提供的轻量级模型
	
	例如:
	def captcha_model():
	inputs = tf.keras.layers.Input(shape=inputs_shape)
	x = Densenet.Densenet(inputs, num_init_features=64, growth_rate=32, block_layers=[6, 12, 32, 32],
						  compression_rate=0.5,
						  drop_rate=0.5)
	outputs = tf.keras.layers.Dense(units=CAPTCHA_LENGTH * Settings.settings(),
									activation=tf.keras.activations.softmax)(x)
	outputs = tf.keras.layers.Reshape((CAPTCHA_LENGTH, Settings.settings()))(outputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
											   weight_decay=1e-2, rectify=False),
				  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
				  metrics=['acc'])
	return model
	
	查看models.py下面的注释将训练比较慢的模型换成训练比较快的模型
	
	例如:
	def captcha_model():
		inputs = tf.keras.layers.Input(shape=inputs_shape)
		x = Mobilenet.MobileNetV3Small(inputs)
		outputs = tf.keras.layers.Dense(units=CAPTCHA_LENGTH * Settings.settings(),
										activation=tf.keras.activations.softmax)(x)
		outputs = tf.keras.layers.Reshape((CAPTCHA_LENGTH, Settings.settings()))(outputs)
		model = tf.keras.Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
												   weight_decay=1e-2, rectify=False),
					  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
					  metrics=['acc'])
		return model

	注意:模型经过少许魔改可能和大佬原来的网络结构有所不同
	
### 保存训练到一定进度的模型
	很多时候模型训练的差不多了
	
	这时直接停止运行save_model.py
	
	下次又想训练了直接运行train.py
	
	想重头再来手动删除检查点或者运行
	
	init_working_space.py
	
# 3.识别验证码的思路
我们知道输入神经网络都是张量

那么我们看看图片的张量是怎么样子的

tf.Tensor(
[[[[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  ...

  [[0.8980392 ]
   [1.        ]
   [1.        ]
   ...
   [0.90588236]
   [1.        ]
   [0.8901961 ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [0.92156863]
   [1.        ]]

  [[0.9882353 ]
   [0.95686275]
   [1.        ]
   ...
   [0.91764706]
   [1.        ]
   [0.99607843]]]], shape=(1, 40, 100, 1), dtype=float32)

这是经过 本项目 处理后的图片张量的样子
处理方法已经改成如果你设置的高宽比较小，先进行等比缩放，然后在进行填充
大的话直接填充，保证图片不会失真，设置的高宽最好大于数据集的高宽

    def show_image(image):
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        iw, ih = image.size
        w, h = IMAGE_WIDTH, IMAGE_HEIGHT
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        if IMAGE_CHANNALS == 3:
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
        else:
            new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        new_image.show()
		
可以找张图片看下图片会被处理成什么样子
        
已经调整好形状并归一化了

那么标签呢？

首先说说独热编码是怎么回事

例如两个动物猫和狗:

那么表示猫我们用 [1,0]

那么表示狗我们用 [0,1]

这就是独热编码了，为了方便说明和理解我使用数字0到9说明一下标签的处理

例如我们要识别长度为4的验证码 有一张验证码的标签为 5206 那么标签要处理成

[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,  |  0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,  |  1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  |  0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,]
        
为了方便查看我用  |  隔开了 实际中要去掉

可以看到

0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,

表示的就是5，那么其他数字依此类推

那么一张验证码为520的怎么处理呢？

[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,  |  0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,  |  1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  |  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,]

可以看到

0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,

表示为5，多出的0.是表示空白字符

0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,

当我们识别有空白字符时，说明验证码长度不为4，后面把空白字符去掉即可，本项目用'_'代表空白字符

后面就是打包和训练了

CTC的标签比较简单，比如1表示龙，2表示舟

那么龙舟的标签处理成[1,2]

## 关于12306验证码识别的想法

通过抓包可以知道验证码文字部分在图片的上方

验证码图片部分有6张图片且图片的分布是固定的也就是说坐标是固定的

那么可以把图片分割成9份，分开来识别，当然现在只是想想肯定有更好的思路

特别感谢下面一些项目对我的启发

[crnn_by_tensorflow2.2.0](https://github.com/lvjianjin/crnn_by_tensorflow2.2.0)

[安师大教务系统验证码检测](https://github.com/AHNU2019/AHNU_captcha)

[cnn_captcha](https://github.com/nickliqian/cnn_captcha)

[captcha_trainer](https://github.com/kerlomz/captcha_trainer)

[captcha-weibo](https://github.com/skygongque/captcha-weibo/blob/master/client.py)

[lambda-networks](https://github.com/lucidrains/lambda-networks)

[Basic_CNNs_TensorFlow2](https://github.com/calmisential/Basic_CNNs_TensorFlow2)

[captcha_break](https://github.com/ypwhs/captcha_break)

[tf_ResNeSt_RegNet_model](https://github.com/QiaoranC/tf_ResNeSt_RegNet_model)

特别说明一下由于大佬的代码不装pytorch是装不上的

所以我直接复制到了models.py里面如有不妥之处请及时联系我删除

@inproceedings{
    anonymous2021lambdanetworks,
    title={LambdaNetworks: Modeling long-range Interactions without Attention},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=xTJEN-ggl1b},
    note={under review}
}

[大佬的优化器](https://github.com/juntang-zhuang/Adabelief-Optimizer)

@article{zhuang2020adabelief,
  title={AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients},
  author={Zhuang, Juntang and Tang, Tommy and Ding, Yifan and Tatikonda, Sekhar and Dvornek, Nicha and Papademetris, Xenophon and Duncan, James},
  journal={Conference on Neural Information Processing Systems},
  year={2020}
}

特别感谢这个大佬
[bubbliiiing](https://github.com/bubbliiiing/efficientdet-tf2)

由于大佬的设置比较麻烦，我做了少许修改如有不妥之处请及时联系我删除

感谢大佬们的数据集让我省去很多成本和时间

搜狗验证码链接： 提取码：9uxv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-149.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

微博验证码链接： 提取码: 74uv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-470.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

12306验证码链接： 提取码：e89o
-------------------------------------
作者: sml2h3
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-84.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

目标检测的数据集获取方式:

(generate_click_captcha)[https://github.com/yuzhiyizhan/generate_click_captcha]
(behavior_captcha_cracker)[https://github.com/eddylapis/behavior_captcha_cracker]

### ***此项目以研究学习为目的，禁止用于非法用途***
### 再次说明项目的tensorflow的版本是(2.2)(2.3)不要搞错了


### 模型保存在分支,与大家共同学习(建议自己训练)

### ps:新手上路，轻喷
### 如果觉得我写的不好或者想教我CRNN + CTC 或者有不懂的地方

### 加我的微信

qq2387301977
 
### 备注熊猫验证

# 更新日志

## 2020/08/09

    微博加搜狗验证码识别率99.75%
    
    12306图片识别率99.46%

    待更新12306文字
   
## 2020/08/10
    
    12306文字识别率99.7%
    
    待更新整合api
    
## 2020/08/11
    
    整合API
    
    添加MODE设置
    'ordinary'      微博加搜狗
    'n_class'       12306图片
    'ordinary_ocr'  12306文字
    
    待更新模型部署
	
## 2020/09/13

	1.MODE更换成：'ORDINARY'，'NUM_CLASSES'，'CTC'

	2.取消用内置函数ord()形成映射表，运行pack_dataset.py的时候
	自动生成num_classes.json，映射表

	3.取消直接对图片resize，这样图片可能会失真，改成填充(不足设置的高宽进行补0)
	
	4.增加inception，densenet，efficientnet等CNN模型
	十分推荐Densenet_169，本人将微博验证码，搜狗验证码，12306_top
	一起训练正确率也达到了86%(一轮训练要一个小时我只练了4轮，多训练几次达到98%以上都是可能的)
	
	5.旧的项目移动至olded分支，模型太大有50M左右
	不会再放模型在主分支
	
	6.由于显卡太垃圾所以CTC暂时还运行不起来，不过流程是没问题的
	用CPU可以训练但是训练太过于慢，对自己硬件自信的朋友可以试下
	后面假如中了彩票的话，就新建一个分支放CTC的模型，识别通用文字
	
	7.待更新模型部署
	
## 2020/09/20

	使用标签平滑提升准确率，降低过拟合(防止模型太膨胀)
	
	原本搜狗验证码只有93.85%正确率(训练6轮)，使用标签平滑后达到96.14%(训练7轮)
	
	待更新模型部署
	
## 2020/12/4

	删除两个作用较小的文件
	
	添加模型部署的方法
	
	12306识别验证码思路是正确的
	
	各位自己切图识别即可
	
	待更新目标检测轻量模型，测试MAP
	
	
    
# 遇到的错误和解决方法

## 错误一:
验证的正确率很不错，测试时全错，或者正确率非常非常低

## 解决方法:
模型有问题，推荐使用Densenet_169，ResNeXt101，SEResNet152
模型在models.py的注释里面(注意，模型经过我的魔改)

## 错误二:
CUDNN_STATUS_INTERNAL_ERROR
显存不足

## 解决方法:
将占显存的地方放到CPU运行
减低模型的复杂程度，降低要训练的参数

## 错误三:
Failed to call ThenRnnForward

## 解决方法:
将BATCH_SIZE调小一点

## 错误四:
训练时损失降低至0点几，评估函数acc降低至0然后报错 nan

## 解决方法:
停止训练，将学习率调小

## 错误五:
一开始损失为nan

## 解决方法:
停止训练，检查数据，网络结构，网络激活函数等

## 错误六:
刚开始损失在降，后来变成nan

## 解决方法:
停止训练，降低学习率LR

## 错误七
loss和acc双降

## 解决方法
稍微等几轮，如果acc还在降，调小学习率

## 错误七
RuntimeWarning: Mean of empty slice

RuntimeWarning: invalid value encountered in true_divide
  ret, rcount, out=ret, casting='unsafe', subok=False)
  
## 解决方法
停止再运行，直到这个错误消失

## 错误八
"Physical devices cannot be modified after being initialized")
RuntimeError: Physical devices cannot be modified after being initialized
    
## 解决方法
注释train.py下的
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    tf.print(gpu)
    
## 错误九
TF2tensorflow/stream_executor/cuda/cuda_dnn.cc:328] 
Could not create cudnn handle: CUDNN_STATUS_IN
    
## 解决方法
开头加入：
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## 错误十
"Physical devices cannot be modified after being initialized")
RuntimeError: Physical devices cannot be modified after being initialized

## 解决方法
重启设备

## 错误十一
OSError: Unable to open file (bad object header version number)

## 解决方法
检查点(.hdf5)文件并没有保存成功，删掉即可
