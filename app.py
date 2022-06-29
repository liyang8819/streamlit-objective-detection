import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import shutil
import logging


def label(img_dir, labels):
    st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager(img_dir)

    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])
    
    def refresh():
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')
        
    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')

    def next_annotate_file():
        image_index = st.session_state["image_index"]
        next_image_index = idm.get_next_annotation_image(image_index)
        if next_image_index:
            st.session_state["image_index"] = idm.get_next_annotation_image(image_index)
        else:
            st.warning("All images are annotated.")
            next_image()
            
                       

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index
        
    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])
    col1,col2,col3=st.columns(3)
    
    col1.write("全部图片:  "+str(n_files))
    col2.write("已标注图片:  "+str(n_annotate_files))
    col3.write("未标注图片:  "+str( n_files - n_annotate_files))

    col1, col2, col3, col4= st.columns([2,2,2,2])
    current_img=col1.selectbox(
                            "Files",
                            st.session_state["files"],
                            index=st.session_state["image_index"],
                            on_change=go_to_image,
                            key="file",
                        )   
    
    with col2:
        st.write("*")
        st.button(label="刷新文件夹", on_click=refresh)
        
    with col3:    
        model_ok,model_num=model_sel()           
        if model_ok:
            with open('./config/model_sel_config.txt','w', encoding='utf-8') as f:
                 f.write(model_num)
        if not model_ok:
            st.caption("*")
            st.caption('当前无训练好的模型')  
           
    with col4: 
       col4.write("*")         
       if col4.button("全部自动标注"):
           st.info('自动标注开始')
           auto_label_allunlabeled(img_dir)
           st.info('标注成功,请刷新文件夹')
           

    # Main content: annotate images
    img_file_name = idm.get_image(st.session_state["image_index"])
    img_path = os.path.join(img_dir, img_file_name)

    im = ImageManager(img_path)
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
    rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

    def annotate():
        im.save_annotation()
        image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        next_annotate_file()
        
        
        
    col1, col2 , col3, col4, col5= st.columns([1,1,2,2,2])      
    with col1:
        st.button(label="上一张", on_click=previous_image)        
    with col2:        
        st.button(label="下一张", on_click=next_image)  
    with col3:
        st.button(label="未标注的下一张", on_click=next_annotate_file)
    with col4: 
       if col4.button("自动标注"):
           st.info('自动标注开始')
           auto_label(img_dir,current_img)
           st.success('自动标注成功')
           st.button('刷新')                                 
    with col5:           
        st.button(label="保存标注", on_click=annotate)                         
    preview_imgs = im.init_annotation(rects)  
    for i, prev_img in enumerate(preview_imgs):
        prev_img[0].thumbnail((200, 200))
        col1, col2 = st.columns(2)
        with col1:
            col1.image(prev_img[0])
        with col2:
            default_index = 0
            if prev_img[1]:
                default_index = labels.index(prev_img[1])

            select_label = col2.selectbox(
                "Label", labels, key=f"label_{i}", index=default_index
            )
            im.set_annotation(i, select_label)
    
    return current_img
    
def config_task():
    st.Files()
    
def get_config(path='./config/label_config.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        label_exist=f.readlines()[0]  
    return label_exist

def model_sel():
    models=os.listdir('./yolov5_master/runs/train')
    if len(models)>=1:
        model_num=st.selectbox('选择自动标注模型：', models)
        model_ok=1
    else:        
        model_ok=0
        model_num=None           
    return model_ok,model_num
    
    

def auto_label(img_dir,current_img):
    
        shutil.copy('./'+img_dir+'/'+current_img, './'+'yolov5_master/data/images_detect/')
        os.system("python ./yolov5_master/detect.py --save-txt")  
        run_dirlist=os.listdir(r'./yolov5_master/runs/detect')
        exp_num=str(max([int(x[3:]) if x[3:]!="" else 0 for x in run_dirlist]))    
        
        shutil.copy('./yolov5_master/runs/detect/exp'+exp_num+'/'+current_img,
                    './yolov5_master/runs/detect/exp'+exp_num+'/labels')
        
        file_dir = './yolov5_master/runs/detect/exp'+exp_num+'/labels'
        lists=[]
        for i in os.listdir(file_dir):
            if i[-3:]=='jpg':
                lists.append(file_dir+'/'+i) 
        translate(file_dir,lists)
        shutil.copy('./yolov5_master/runs/detect/exp'+exp_num+'/labels/'+current_img[0:-3]+'xml',img_dir) 
    
def auto_label_allunlabeled(img_dir):
    idm = ImageDirManager(img_dir)
    all_pic= idm.get_all_files()
    xml_files = idm.get_exist_annotation_files()
    labeled_pic=[x.strip('.xml')+'.jpg' for x in xml_files]
    unlabeled_pic=set(all_pic)-set(labeled_pic)

    for unlabeled_picx in unlabeled_pic:
        shutil.copy('./'+img_dir+'/'+unlabeled_picx, './'+'yolov5_master/data/images_detect/')
    os.system("python ./yolov5_master/detect.py --save-txt")  
    run_dirlist=os.listdir(r'./yolov5_master/runs/detect')
    exp_num=str(max([int(x[3:]) if x[3:]!="" else 0 for x in run_dirlist]))  
    
    for unlabeled_picx in unlabeled_pic:
        shutil.copy('./yolov5_master/runs/detect/exp'+exp_num+'/'+unlabeled_picx,
                    './yolov5_master/runs/detect/exp'+exp_num+'/labels')
    
    file_dir = './yolov5_master/runs/detect/exp'+exp_num+'/labels'
    lists=[]
    for i in os.listdir(file_dir):
        if i[-3:]=='jpg':
            lists.append(file_dir+'/'+i) 
    translate(file_dir,lists)
    for unlabeled_picx in unlabeled_pic:
        shutil.copy('./yolov5_master/runs/detect/exp'+exp_num+'/labels/'+unlabeled_picx[0:-3]+'xml',img_dir)                        

def gpu_info():        
    import torch
    import torchvision
    st.write('torch版本号:',torch.__version__)
    st.write('cuda是否可用:',torch.cuda.is_available())
    st.write('torchvision版本号:',torchvision.__version__)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.write('设备:',device)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            st.write('第'+str(i)+'块gpu设备名:',torch.cuda.get_device_name(i))
        st.write('cuda版本号:',torch.version.cuda)
        st.write('cudnn版本号:',torch.backends.cudnn.version())
        st.write('cuda计算测试:',torch.rand(3,3).cuda()) 
        
def cp_tree_ext(exts,src,dest):
  """
  Rebuild the director tree like src below dest and copy all files like XXX.exts to dest 
  exts:exetens seperate by blank like "jpg png gif"
  """
  fp={}
  extss=exts.lower().split()
  for dn,dns,fns in os.walk(src):
    for fl in fns:
      if os.path.splitext(fl.lower())[1][1:] in extss:
        if dn not in fp.keys():
          fp[dn]=[]
        fp[dn].append(fl)
  for k,v in fp.items():
      relativepath=k[len(src)+1:]
      newpath=os.path.join(dest,relativepath)
      for f in v:
        oldfile=os.path.join(k,f)
        print("拷贝 ["+oldfile+"] 至 ["+newpath+"]")
        if not os.path.exists(newpath):
          os.makedirs(newpath)
        shutil.copy(oldfile,newpath)

def train_model(img_dir):
    
     
    import yaml
    import os
    
    # 训练集拷贝到特定的配置文件夹-----------    
    shutil.rmtree('./yolov5_master/data/Annotations/')
    shutil.rmtree('./yolov5_master/data/images/')
    cp_tree_ext('xml',img_dir,'./yolov5_master/data/Annotations/')
    cp_tree_ext('jpg',img_dir,'./yolov5_master/data/images/')

    
    # 进入yolov5_master文件夹，生成配置文件-----------
    os.chdir(os.getcwd()+'/yolov5_master')    
    os.system("python makeTxt.py")  
    os.system("python voc_label.py")      
    with open('./data/ori.yaml','r',encoding='utf8') as file:
        config_yaml=yaml.safe_load(file)
        
    with open('../config/label_config.txt', 'r', encoding='utf-8') as f:
        label_exist=list(f.readlines()[0].split(',')) 
        classes=label_exist
    
    config_yaml['nc']=len(classes)
    config_yaml['names']=classes 
    name_yaml='_'.join(str(x) for x in classes)
    with open('./data/'+name_yaml+'.yaml','w') as f:
        yaml.dump(config_yaml,f)    
    

    # 训练--------------------------------
    with st.spinner('训练中...'):
        st.caption("训练进度")
        from train import main
        from argparse_opt import parse_opt
        opt=parse_opt()
        main(opt)
        os.chdir(os.getcwd().strip('/yolov5_master')) 
         
    
def train_process():        
    # if st.button("训练结果"):   
    os.system('activate yolospyder')
    import webbrowser
    os.system('tensorboard --logdir D:/ly/streamlit-objective-detection/yolov5_master/runs/train/'+expname) 
    webbrowser.open('http://localhost:6006/', new=0, autoraise=True) 
    
def get_model_training_logs(expname,n_lines = 1):
    file = open('./yolov5_master/'+expname+'.log', 'r')
    lines = file.read().splitlines()
    file.close()
    return lines[-n_lines:]    
    
    
if __name__ == "__main__":
    
    st.set_page_config(page_title="HCE图像目标平台", 
                        page_icon="random" , 
                        # layout="wide",
                        initial_sidebar_state="auto")
    
    st.markdown(""" <style> .font3 {
    font-size:20px ; font-family: 'Cooper Black'; color: red;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:25px ; font-family: 'Cooper Black'; color: #red;} 
    </style> """, unsafe_allow_html=True)
    
    from PIL import Image
    image = Image.open('./logo/Honeywell_Logo_RGB_Red.png')    
    col1,col2=st.columns([5,7])
    col1.image(image, width=300) 
    col2.write("")
    col2.write("")
    col2.markdown('<p class="font3">企业智联 | 目标检测助手</p>', unsafe_allow_html=True)

    # st.sidebar.image('./image/honeywell.png',width=150)
    
    from txt2xml import translate
    
    with st.sidebar:    
        choose = option_menu("功能栏", ["图片标注","训练模型"],                             
                             icons=[
                                    'gear-fill',
                                    'plus-lg'], 
                             menu_icon="list", default_index=0)
    if choose =="图片标注":
        i=0
        label_task = stx.stepper_bar(steps=["Step1:配置", "Step2:标注", "Step3:增强"], is_vertical=0, lock_sequence=False)
        
        if label_task==0:
            # st.header("step1，图片路径")

            img_dir_prev=get_config(path='./config/imgpath_config.txt')  
            img_dir=st.text_input('图片路径',value=img_dir_prev) 
 
                
                
            # st.header("step2，创建Label")
            st.write("")
            st.write("")
            st.write("")
            label_exist=get_config()               
            label_new=st.text_input('创建label',value=label_exist) 
            st.write('当前label :   ',label_new)
            st.write("")
            st.write("")
            st.write("") 
            
            col1,col2=st.columns([7,1])
            config_button=col2.button("确认")            
            if config_button and img_dir!="":
                os.remove('./config/imgpath_config.txt')
                with open('./config/imgpath_config.txt','w', encoding='utf-8') as f:
                    f.write(img_dir)
                st.info('创建路径 '+str(img_dir)+' 成功')
            if config_button and label_new!="":
                os.remove('./config/label_config.txt')
                with open('./config/label_config.txt','w', encoding='utf-8') as f:
                    f.write(label_new) 
                st.info('创建label '+str(label_new)+' 成功')    
                


                                                                          

        if label_task==1:
            img_dir=get_config(path='./config/imgpath_config.txt')
            labels=get_config().split(',')
            current_img=label(img_dir,labels)
        
    if choose =="训练模型": 
        
        # '''part 1 设备信息'''       
        st.write("设备信息：")
        with st.expander("-："):
            gpu_info() 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        

        # '''part 2 图片和标注文件的路径'''                        
        img_dir_prev=get_config(path='./config/imgpath_config.txt') 
        
        img_dir=st.text_input('训练集路径：',value=img_dir_prev) 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
    
        
        # '''part 3 模型命名'''         
        expname=st.text_input('模型命名：') 

        if len(expname)>0:
            with open('./config/train_name.txt','w', encoding='utf-8') as f:
                f.write(expname)   
                
                
        # '''part 4 训练相关按钮'''                                                                
        col1,col2,col3=st.columns(3)
        with col1:
            start=st.button("开始训练",key='start')


        # '''part 5 训练'''           
        if start and len(expname)==0: 
            st.warning('请先对模型命名')                                        
        if start and len(expname)>0: 
            train_model(img_dir)
            st.success("训练完成")

        # '''part 5 训练结果''' 
 
        import webbrowser
        import threading
        def openurl():
            webbrowser.open('http://localhost:6006/', new=0, autoraise=True)
        def tensorboard_():
            os.system('tensorboard --logdir D:/ly/streamlit-objective-detection/yolov5_master/runs/train/'+exp) 
                              
        
        col1,col2,col3=st.columns([3,1,1])
        exp=col2.selectbox('选择模型', os.listdir('./yolov5_master/runs/train'))

        with col3:
            st.write("*")
            if st.button('查看训练结果',key='result'):
                os.system('activate yolospyder') 
                    
                t1=threading.Thread(target=tensorboard_)
                t2=threading.Thread(target=openurl)
                t1.start()
                t2.start()