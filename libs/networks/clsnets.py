import os
import numpy as np
import scipy.special
import tensorflow as tf
import matplotlib.pyplot as plt
from libs.commons.logs import Logs
from libs.networks.losses import Losses
from libs.datasets.dataset import CSDataset
from libs.networks.callbacks import CustomCallback
from libs.networks.lrs import LearningRateScheduler
class ClsNets:
    def __init__(self,
                 i_model_name  = 'VGG16',
                 i_image_shape = (256,256,3),
                 i_time_steps  = 1,
                 i_fine_tune   = True,
                 i_num_classes = 2):
        assert isinstance(i_model_name,str)
        assert isinstance(i_image_shape,(list,tuple))
        assert isinstance(i_time_steps,int)
        assert isinstance(i_fine_tune,bool)
        assert isinstance(i_num_classes,int)
        assert len(i_image_shape)==3
        self.model_name  = i_model_name
        self.image_shape = i_image_shape
        self.time_step   = i_time_steps
        self.fine_tune   = i_fine_tune
        self.num_classes = i_num_classes
        """Calculate the input shape"""
        if self.time_step == 1:
            self.input_shape = self.image_shape
        else:
            self.input_shape = (self.time_step,) + self.image_shape
    """Build the model"""
    def build(self):
        input_shape = (self.image_shape[0],self.image_shape[1],3) #Input shape of popular nets
        if self.model_name == 'VGG16':
            model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'ResNet101':
            model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'ResNet152':
            model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'Inception':
            model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'InceptionResNet':
            model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == "DenseNet121":
            model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == "DenseNet169":
            model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == "DenseNet201":
            model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape)
        elif self.model_name == 'cCNN':
            return ClsNets.custom_cnn(i_input_shape=self.image_shape,i_filters=32,i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name  == 'cCNN_MS':
            return  ClsNets.custom_cnn(i_input_shape=self.image_shape, i_filters=32, i_multiscale=True,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny':
            return  ClsNets.custom_cnn_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny_MS':
            return  ClsNets.custom_cnn_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=True,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny_tiny':
            return ClsNets.custom_cnn_tiny_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=False,i_num_classes=self.num_classes)
        elif self.model_name == 'cCNN_tiny_tiny_MS':
            return ClsNets.custom_cnn_tiny_tiny(i_input_shape=self.image_shape, i_filters=32, i_multiscale=True,i_num_classes=self.num_classes)
        else:#Default is VGG16
            model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        """Contruct model"""
        model.trainable = False if self.fine_tune else True
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        if self.input_shape[-1]!=3:
            """Making color image from gray image to be used in conventional classification network"""
            outputs = tf.keras.layers.Conv2D(filters=3,kernel_size=(3,3),strides=(1,1),padding='same')(inputs)
        else:
            outputs = inputs
        if self.time_step==1:
            """Conventional classification networks"""
            outputs = model(outputs)
            """2. Futher manipulation"""
            if self.fine_tune:
                outputs = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),activation='relu')(outputs)
            else:
                pass
            outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
            outputs = tf.keras.layers.Flatten()(outputs)
        else:
            """Classification networks for sequence of images"""
            outputs = tf.keras.layers.TimeDistributed(model)(outputs)
            """2. Futher manipulation"""
            if self.fine_tune:
                outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))(outputs)
            else:
                pass
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(outputs)
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(outputs)
            """Combine feature"""
            outputs = tf.keras.layers.Flatten()(outputs)
        """Classification layers"""
        outputs = tf.keras.layers.Dense(units=self.num_classes, activation=None, name='combine')(outputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    """Custom network"""
    @staticmethod
    def residual_block(i_inputs, i_kernel_size, i_nb_filters, i_stride=(1, 1), use_bias=True):
        """Custom network using residual connection blocks"""
        assert isinstance(i_kernel_size, int)
        assert isinstance(i_nb_filters, (list, tuple))
        assert len(i_nb_filters) == 3
        assert i_kernel_size >= 3
        nb_filter1, nb_filter2, nb_filter3 = i_nb_filters
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias)(i_inputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter2, kernel_size=(i_kernel_size, i_kernel_size),strides=i_stride, padding='same', use_bias=use_bias)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        """Shortcut CONV block"""
        shortcut = tf.keras.layers.Conv2D(nb_filter3, (3, 3), strides=i_stride, padding='same', use_bias=use_bias)(i_inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        """Aggregation"""
        outputs = tf.keras.layers.Add()([outputs, shortcut])
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @staticmethod
    def custom_cnn(i_input_shape, i_filters=32, i_multiscale=False, i_num_classes=2):
        assert isinstance(i_num_classes, int)
        assert i_num_classes > 0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 1
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 1
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(tf.keras.layers.MaxPool2D()(outputs))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(1, 1))
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, 4 * filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    @staticmethod
    def custom_cnn_tiny(i_input_shape, i_filters=32, i_multiscale=False, i_num_classes=2):
        assert isinstance(i_num_classes, int)
        assert i_num_classes > 0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 1
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 1
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(tf.keras.layers.MaxPool2D()(outputs))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, 4 * filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    @staticmethod
    def custom_cnn_tiny_tiny(i_input_shape, i_filters=32, i_multiscale=False, i_num_classes=2):
        assert isinstance(i_num_classes, int)
        assert i_num_classes > 0
        ms_outputs = []
        inputs = tf.keras.layers.Input(shape=i_input_shape)
        """1. Warming-up layer"""
        outputs = tf.keras.layers.Conv2D(filters=i_filters, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu')(inputs)
        """2. Residual connection layers"""
        filters = i_filters * 1
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 2
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        filters = i_filters * 4
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(tf.keras.layers.MaxPool2D()(outputs))
        filters = i_filters * 8
        outputs = ClsNets.residual_block(i_inputs=outputs, i_kernel_size=3,i_nb_filters=(filters, 2 * filters, filters), i_stride=(2, 2))
        ms_outputs.append(outputs)
        if i_multiscale:
            for index, outputs in enumerate(ms_outputs):
                ms_outputs[index] = tf.keras.layers.Flatten()(outputs)
            ms_outputs = tf.keras.layers.Concatenate(axis=-1)(ms_outputs)
            ms_outputs = tf.keras.layers.Dropout(rate=0.250)(ms_outputs)
            ms_outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(ms_outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[ms_outputs])
        else:
            outputs = tf.keras.layers.Flatten()(outputs)
            outputs = tf.keras.layers.Dropout(rate=0.250)(outputs)
            outputs = tf.keras.layers.Dense(units=i_num_classes, name='output')(outputs)
            return tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
class ImgClsNets:
    cv_debug = False
    def __init__(self,
                 i_net_id      = 0,
                 i_save_path   = None,
                 i_image_shape = (256,256,3),
                 i_num_classes = 2,
                 i_continue    = True,**kwargs):
        assert isinstance(i_net_id,int)
        assert i_net_id>=0
        assert isinstance(i_image_shape,(list,tuple))
        assert len(i_image_shape)==3
        assert isinstance(i_num_classes,int)
        assert i_num_classes>1
        assert isinstance(i_continue,bool)
        self.net_id      = i_net_id
        self.input_shape = i_image_shape
        self.num_classes = i_num_classes
        self.tcontinue   = i_continue
        """Init the save path to save model"""
        if i_save_path is None:
            self.save_path = os.path.join(os.getcwd(), 'ckpts')
        else:
            assert isinstance(i_save_path, str)
            self.save_path = i_save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        self.model_path = os.path.join(self.save_path, 'clsnet_{}.h5'.format(self.net_id))
        """Init the model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(filepath=self.model_path,custom_objects=Losses.get_custom_objects())
        else:
            self.model = None
        """Inference training parameters"""
        if 'cls_lr' in kwargs.keys():
            self.lr = kwargs['cls_lr']
        else:
            self.lr = 0.0001
        if 'cls_loss' in kwargs.keys():
            self.loss = kwargs['cls_loss']
        else:
            self.loss = 'CE'
        if 'cls_repeat' in kwargs.keys():
            self.db_repeat = kwargs['cls_repeat']
        else:
            self.db_repeat = 1
        if 'cls_epochs' in kwargs.keys():
            self.num_epochs = kwargs['cls_epochs']
        else:
            self.num_epochs = 10
        if 'cls_weight' in kwargs.keys():
            self.weight = kwargs['cls_weight']
        else:
            self.weight = [0.25, 0.75]
        if 'cls_lsm_factor' in kwargs.keys():
            self.lsm_factor = kwargs['cls_lsm_factor']
        else:
            self.lsm_factor = 0.0
        if 'cls_batch_size' in kwargs.keys():
            self.batch_size = kwargs['cls_batch_size']
        else:
            self.batch_size = 16
        if 'cls_ori_output' in kwargs.keys():
            self.get_original_output = kwargs['cls_ori_output']
        else:
            self.get_original_output = False
        if 'cls_time_step' in kwargs.keys():
            self.time_step = kwargs['cls_time_step']
        else:
            self.time_step = 1
        if 'cls_fine_tune' in kwargs.keys():
            self.fine_tune = kwargs['cls_fine_tune']
        else:
            self.fine_tune = False
        assert isinstance(self.fine_tune,bool)
        if 'cls_flip_ud' in kwargs.keys():
            self.flip_ud = kwargs['cls_flip_ud']
        else:
            self.flip_ud = False
        assert isinstance(self.flip_ud,bool)
        if 'cls_flip_lr' in kwargs.keys():
            self.flip_lr = kwargs['cls_flip_lr']
        else:
            self.flip_lr = False
        assert isinstance(self.flip_lr,bool)
        if 'cls_crop_size' in kwargs.keys():
            self.crop_size = kwargs['cls_crop_size']
        else:
            crop_height = int(self.input_shape[0]*0.9)
            crop_width  = int(self.input_shape[1]*0.9)
            self.crop_size = (crop_height,crop_width,self.input_shape[2])
        if 'cls_threshold' in kwargs.keys():
            self.threshold = kwargs['cls_threshold']
        else:
            if self.num_classes==1:
                self.threshold = 0.5
            elif self.num_classes==2:
                self.threshold = 0.0
            else:
                self.threshold = 0.5
        assert isinstance(self.crop_size,(list,tuple))
        assert len(self.crop_size)==3
        assert self.crop_size[0]<=self.input_shape[0]
        assert self.crop_size[1]<=self.input_shape[1]
        assert self.crop_size[2]==self.input_shape[2]
    def init_network(self):
        if self.net_id==0:#VGG16
            return ClsNets(i_model_name  = 'VGG16',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==1:#VGG19
            return ClsNets(i_model_name  = 'VGG19',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==2:#ResNet50
            return ClsNets(i_model_name  = 'ResNet50',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==3:#Inception
            return ClsNets(i_model_name  = 'Inception',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==4:#InceptionResnet
            return ClsNets(i_model_name  = 'InceptionResNet',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==5:#DenseNet
            return ClsNets(i_model_name  = 'DenseNet121',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==6:#DenseNet
            return ClsNets(i_model_name  = 'cCNN',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==7:#DenseNet
            return ClsNets(i_model_name  = 'cCNN_tiny',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==8:#DenseNet
            return ClsNets(i_model_name  = 'cCNN_tiny_tiny',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==9:#DenseNet
            return ClsNets(i_model_name  = 'cCNN_MS',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==10:#DenseNet
            return ClsNets(i_model_name  = 'cCNN_tiny_MS',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        elif self.net_id==11:#DenseNet
            return ClsNets(i_model_name  = 'cCNN_tiny_tiny_MS',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
        else:
            return ClsNets(i_model_name  = 'VGG16',
                           i_image_shape = self.input_shape,
                           i_time_steps  = self.time_step,
                           i_num_classes = self.num_classes,
                           i_fine_tune   = self.fine_tune).build()
    @staticmethod
    def pipeline(i_record=None,i_ori_shape=(256,256,1),i_lsm_factor=0.0,i_num_classes=2,i_train_flag=True,**kwargs):
        """Pipeline the data record obtained from tfrecord dataset. Check the CSDataset object for more detail"""
        """i_record is the output of tf.data.Dataset after doing pipeline()"""
        """i_ori_shape is the original shape of input image"""
        assert isinstance(i_record, (list, tuple, dict))
        assert isinstance(i_ori_shape,(list,tuple))
        assert len(i_ori_shape)==3
        assert isinstance(i_train_flag, bool)
        if isinstance(i_record, (list, tuple)):
            image, label = i_record
        else:
            image, label = i_record['image'], i_record['label']
        assert isinstance(image, (tf.Tensor, tf.SparseTensor))
        assert isinstance(label, (tf.Tensor, tf.SparseTensor))
        assert image.dtype in (tf.dtypes.uint8,)
        assert label.dtype in (tf.dtypes.uint8,tf.dtypes.int32,tf.dtypes.int64)
        image = tf.reshape(tensor=image, shape=i_ori_shape)
        if 'flip_ud' in kwargs.keys():
            flip_ud = kwargs['flip_ud']
        else:
            flip_ud = False
        if 'flip_lr' in kwargs.keys():
            flip_lr = kwargs['flip_lr']
        else:
            flip_lr = False
        if 'crop_size' in kwargs.keys():
            crop_size = kwargs['crop_size']
        else:
            crop_size = i_ori_shape
        if i_train_flag:
            if flip_ud:
                image = tf.image.random_flip_up_down(image)
            else:
                pass
            if flip_lr:
                image = tf.image.random_flip_left_right(image)
            else:
                pass
            image = tf.image.random_crop(value=image, size=crop_size)
            image = tf.image.resize(image, size=i_ori_shape[0:2])
            """Normalization"""
            image = tf.cast(image, tf.dtypes.float32) / 255.0
            """Label smoothing"""
            label = tf.cast(tf.one_hot(label, i_num_classes), tf.dtypes.float32)
            label = label * (1.0 - i_lsm_factor)
            label = label + i_lsm_factor / i_num_classes
        else:
            """Normalization"""
            image = tf.cast(image, tf.dtypes.float32) / 255.0
            label = tf.cast(tf.one_hot(label, i_num_classes), tf.float32)
        return image, label
    def train(self,i_train_db=None,i_val_db=None):
        """i_train_db can be (list, tuple, None)"""
        """i_val_db can be (list, tuple, None)"""
        """If i_train_db is None,then we will read the saved tfrecord file at the self.save_path location."""
        """If i_val_db is None,then we will read the saved tfrecord file at the self.save_path location."""
        db       = CSDataset(i_save_path=self.save_path, i_target_shape=self.input_shape)
        train_db = db.prepare(i_db=i_train_db, i_train_flag=True, i_lsm_factor=self.lsm_factor,i_num_classes=self.num_classes,
                              i_pipeline_fn=self.pipeline,flip_ud=self.flip_ud,flip_lr=self.flip_lr,crop_size=self.crop_size)
        val_db   = db.prepare(i_db=i_val_db, i_train_flag=False, i_lsm_factor=self.lsm_factor,i_num_classes=self.num_classes,
                              i_pipeline_fn=self.pipeline)
        train_db = train_db.batch(self.batch_size)
        val_db   = val_db.batch(self.batch_size)
        if self.cv_debug:
            for batch in val_db:#train_db
                debug_images,debug_labels = batch
                for index, image in enumerate(debug_images):
                    label = tf.argmax(debug_labels[index],axis=-1)
                    plt.imshow(image, cmap='gray')
                    plt.title('Image - {}'.format(label))
                    plt.show()
                    if index>3:
                        break
                    else:
                        pass
                break
        else:
            pass
        """Model initialization"""
        if os.path.exists(self.model_path):
            if self.tcontinue:
                network = tf.keras.models.load_model(filepath=self.model_path,custom_objects=Losses.get_custom_objects())
            else:
                self.eval(i_db=train_db)
                self.eval(i_db=val_db)
                return False
        else:
            Logs.log("Train a new model from scratch")
            network = self.init_network()
        assert isinstance(network, tf.keras.models.Model)
        network.summary()
        net = Losses.compile(i_net=network, i_lr=self.lr, i_loss_name=self.loss,i_weights=self.weight)
        """Training"""
        log_infor  = CustomCallback(i_model_path=self.model_path)
        lr_schuler = LearningRateScheduler()
        lr_params  = {'decay_rule': 1, 'step': int(self.num_epochs / 10), 'decay_rate': 0.90, 'base_lr': self.lr}
        schedule   = lr_schuler(lr_params)
        callbacks  = [schedule, log_infor]
        network.fit(x               = train_db.repeat(self.db_repeat),
                    epochs          = self.num_epochs,
                    verbose         = 1,
                    shuffle         = True,
                    validation_data = val_db,
                    callbacks       = callbacks)
        """Update the nework"""
        self.model = tf.keras.models.load_model(self.model_path, custom_objects=Losses.get_custom_objects())
        return net
    def predict(self,i_image=None):
        assert isinstance(i_image,np.ndarray)
        assert len(i_image.shape) in (2, 3, 4)
        """Make image batch"""
        image_shape = i_image.shape
        if len(image_shape) in (2, 3):
            if len(image_shape) == 2:
                images = np.expand_dims(i_image, axis=-1)
            else:
                images = i_image.copy()
            images = np.expand_dims(images, axis=0)
        else:
            images = i_image.copy()
        assert images.shape[-1] in (1, 3)
        """Size and Color adjustment"""
        norm_images = []
        for image in images:
            assert isinstance(image, np.ndarray)
            """Color adjustment"""
            if image.shape[-1] == self.input_shape[-1]:
                pass
            else:
                if image.shape[-1] == 1:
                    image = np.concatenate((image, image, image), axis=-1)
                else:
                    image = np.mean(image, axis=-1, keepdims=True).astype(image.dtype)
            """Size adjustment"""
            image = CSDataset.imresize(i_image=image, i_tsize=self.input_shape[0:2])
            norm_images.append(image)
        images = np.array(norm_images)
        assert images.dtype in (np.uint8,)
        """Gray level normalization"""
        images = images.astype(np.float) / 255.0
        """Prediction"""
        pred = self.model.predict(images)
        if self.num_classes==1:
            pred = pred > self.threshold
            return  pred.astype(np.int)
        elif self.num_classes==2:
            pred = scipy.special.softmax(pred)
            pred = pred[:, -1] - pred[:, 0]
            pred = pred > self.threshold
            return pred.astype(np.int)
        else:
            return np.argmax(pred,axis=-1)
    """Evaluation function"""
    def eval(self,i_db=None,i_debug=False):
        assert isinstance(i_db, (list, tuple, tf.data.Dataset))
        assert isinstance(i_debug, bool)
        labels, predictions = [], []
        conf_matrix = np.zeros(shape=(self.num_classes,self.num_classes))
        for index, element in enumerate(i_db):
            print("(ImageClsNets) Evaluating element: {}".format(index))
            assert isinstance(element, (list, tuple, dict))
            """Extract data"""
            if isinstance(element, (list, tuple)):
                image, label = element
            else:
                image, label = element['image'], element['label']
            """Preprocess image data"""
            if isinstance(image, (tf.Tensor, tf.SparseTensor)):
                image = image.numpy()
            else:
                if isinstance(image, np.ndarray):
                    pass
                else:
                    assert isinstance(image,(list,tuple))
                    image = np.array(image)
            if len(image.shape) == 2:
                image = np.expand_dims(np.expand_dims(image, -1), 0)
            else:
                if len(image.shape) == 3:
                    if image.shape[-1] in (1, 3):
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = np.expand_dims(image, axis=-1)
                else:
                    pass
            """Label processing"""
            if isinstance(label, (tf.Tensor, tf.SparseTensor)):
                label = label.numpy()
                label = np.argmax(label,axis=-1)
            else:
                if isinstance(label, np.ndarray):
                    pass
                else:
                    if isinstance(label,(list,tuple)):#Multiple image
                        label = np.array(label)
                    else:#Single image
                        assert isinstance(label,(int,np.int,np.uint8))
                        label = np.reshape(label,(1,1))
            """Prediction"""
            preds = self.predict(i_image=image)
            for pred_index, pred in enumerate(preds):
                plabel = label[pred_index]
                conf_matrix[plabel,pred]+=1
                labels.append(plabel)
                predictions.append(pred)
                if self.cv_debug or i_debug:
                    current_image = image[pred_index]
                    plt.imshow(current_image, cmap='gray')
                    plt.title('Image {} vs {}'.format(plabel,pred))
                    plt.show()
                else:
                    pass
        """Performance measurement"""
        Logs.log_matrix('Confusion matrix ',i_matrix=conf_matrix)
        total = np.sum(conf_matrix)
        trace = np.trace(conf_matrix)
        acc   = 100.0*trace/total
        Logs.log('Accuracy = {}/{} ~ {} (%)'.format(trace,total,acc))
        return labels, predictions
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to implement a conventional classification network')
    """=============================================================================================================="""
    #network = ClsNets(i_model_name='VGG16',
    #                  i_time_steps=2,
    #                  i_image_shape=(256, 256, 3),
    #                  i_fine_tune=True,
    #                  i_num_classes=2)
    #clsnet = network.build()
    #clsnet.summary()
    """=============================================================================================================="""
    tr_db, va_db = CSDataset.get_sample_db(i_tsize=(256, 256), i_num_train_samples=10000, i_num_val_samples=1000)
    ImgClsNets.cv_debug = False
    clsnet = ImgClsNets(i_net_id      = 0,
                        i_save_path   = None,
                        i_image_shape = (256, 256, 1),
                        i_num_classes = 10,
                        i_continue    = False)
    tr_db = list(zip(tr_db[0], tr_db[2]))
    va_db = list(zip(va_db[0], va_db[2]))
    clsnet.train(i_train_db=tr_db, i_val_db=va_db)
    clsnet.eval(i_db=tr_db)
    clsnet.eval(i_db=va_db)
    for item in va_db:
        simage, slabel = item
        spred = clsnet.predict(i_image=simage)[0]
        print(spred.shape, np.sum(spred))
        plt.imshow(simage, cmap='gray')
        plt.title('Image - {} vs {}'.format(slabel,spred))
        plt.show()
    """=============================================================================================================="""
"""=================================================================================================================="""