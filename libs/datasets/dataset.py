import os
import imageio
import numpy as np
from PIL import Image
import tensorflow as tf
from libs.datasets.tfrecords import TFRecordDB
"""=====================================================================================================================
- Prepare dataset for a segmentation network
- Inputs: 
    - Classification Problem : List of (image,label) or (path_to_image,label), or mixed of the two
    - Segmentation Problem   : List of (image,mask) or (path_to_image,path_to_mask), or mixed of the two
    - Mixed Problem          : List of (image,mask,label) or (path_to_image,path_to_mask,label), or mixed of the two
    *Note: label must be integer.
- Output: 
    - Save data to tfrecord dataset if NOT exists
    - Read tfrecord dataset and make ready to use by segmentation networks
- Note:
    - image must be gray (h,w) or color (h,w,d) image where d is 1 or 3
    - mask must be gray (h,w) or (h,w,1) image.
    - The bellow class requires a custom pipeline function to be work.
    - The pipeline function is work for a specific problem and must be customed according to application.
====================================================================================================================="""
class CSDataset:#This class used for classification and segmentation networks
    def __init__(self,i_save_path = None,i_target_shape = (256, 256, 3)):
        self.save_path    = i_save_path         #Path to directory where we store tfrecord data
        self.target_shape = i_target_shape      #Shape of images
        self.record_size  = 1000                #Write 10000 (image,mask) pair to a single tfrecord file.
        self.mask_shape   = (self.target_shape[0],self.target_shape[1],1)
    @classmethod
    def imresize(cls, i_image=None, i_tsize=(512, 512), i_min=None, i_max=None):
        """i_tsize in format (height, width). But, PIL.Image.fromarray.resize(width, height). So, we must invert i_tsize before scaling"""
        assert isinstance(i_image, np.ndarray)
        assert isinstance(i_tsize, (list, tuple))
        assert len(i_tsize) == 2
        assert i_tsize[0] > 0
        assert i_tsize[1] > 0
        tsize = (i_tsize[1], i_tsize[0])
        if i_image.dtype != np.uint8:  # Using min-max scaling to make unit8 image
            if i_min is None:
                min_val = np.min(i_image)
            else:
                assert isinstance(i_min, (int, float))
                min_val = i_min
            if i_max is None:
                max_val = np.max(i_image)
            else:
                assert isinstance(i_max, (int, float))
                max_val = i_max
            assert min_val <= max_val
            image = (i_image - min_val) / (max_val - min_val + 1e-10)
            image = (image * 255.0).astype(np.uint8)
        else:
            image = i_image.copy()
        assert image.dtype == np.uint8
        assert isinstance(image,np.ndarray)
        image = Image.fromarray(np.squeeze(image)).resize(tsize)
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        else:
            assert len(image.shape) == 3
        return image
    @classmethod
    def scale_mask(cls, i_mask=None, i_tsize=None):  # Remove i_num_labels later as it affects to some of other files.
        assert isinstance(i_mask, np.ndarray), 'Got type: {}'.format(type(i_mask))
        assert isinstance(i_tsize, (list, tuple, int)), 'Got type: {}'.format(type(i_tsize))
        assert len(i_mask.shape) in (2, 3), 'Got shape: {}'.format(i_mask.shape)
        if isinstance(i_tsize, int):
            assert i_tsize > 0, 'Got value: {}'.format(i_tsize)
            tsize = (i_tsize, i_tsize)
        else:
            assert len(i_tsize) in (2, 3), 'Got values: {}'.format(i_tsize)
            assert 0 < i_tsize[0], 'Got values: {}'.format(i_tsize)
            assert 0 < i_tsize[1], 'Got values: {}'.format(i_tsize)
            tsize = (i_tsize[0], i_tsize[1])
        if len(i_mask.shape) == 2:
            mask = np.expand_dims(i_mask, -1)
        else:
            mask = i_mask.copy()
        assert mask.shape[-1] in (1,)
        num_labels = int(np.max(mask)) + 1
        masks = [np.zeros(shape=(tsize[0], tsize[1], 1), dtype=np.uint8)]
        for index in range(1, num_labels):
            cmask = (mask == index).astype(np.uint8)
            cmask = cls.imresize(i_image=cmask, i_tsize=tsize)
            cmask = (cmask > 0).astype(np.uint8)
            cmask = (cmask * index).astype(np.uint8)
            masks.append(cmask)
        masks = np.concatenate(masks, axis=-1)
        assert len(masks.shape) == 3, 'Got shape: {}'.format(masks.shape)
        assert masks.shape[-1] == num_labels
        mask = np.expand_dims(np.max(masks, axis=-1),axis=-1)  # Selecting the object with small index. Alternative is reduce_max
        return mask  # Return shape: (i_tsize,i_tsize,1)
    def save_tfrecord(self, i_db=None,i_save_path=None):
        assert isinstance(i_db,(list,tuple))
        assert isinstance(i_save_path,str)
        assert not os.path.exists(i_save_path)
        assert len(i_db)>0
        save_path, save_file = os.path.split(i_save_path)
        save_file = save_file[0:len(save_file) - len('.tfrecord')]
        """Object to write data to tfrecord files"""
        TFRecordDB.lossy = False
        tfwriter = TFRecordDB()
        """Object to carry data"""
        images, masks, labels = [], [], []
        count, segment        = 0, 0
        db_fields             = None
        num_fields            = 0
        for index,element in enumerate(i_db):
            assert isinstance(element,(list,tuple))
            assert len(element) in (2,3)
            if len(element)==3:
                image,mask,label = element
                """Init the fields"""
                if db_fields is None:
                    db_fields  = {'image':[],'mask':[],'label':[]}
                    num_fields = 3
                else:
                    if len(element)==num_fields:
                        pass
                    else:
                        raise Exception('Incorrect data format')
            else:
                image,label = element
                if isinstance(label,(int,np.uint8,np.int)):
                    mask = np.zeros(self.mask_shape)
                    """Init the fields"""
                    if db_fields is None:
                        db_fields  = {'image': [],'label': []}
                        num_fields = 2
                    else:
                        if len(element) == num_fields:
                            pass
                        else:
                            raise Exception('Incorrect data format')
                else:
                    assert isinstance(label, (np.ndarray, str))
                    mask  = label.copy()
                    label = 0
                    """Init the fields"""
                    if db_fields is None:
                        db_fields  = {'image': [], 'mask': []}
                        num_fields = 2
                    else:
                        if len(element) == num_fields:
                            pass
                        else:
                            raise Exception('Incorrect data format')
            """Note: We will not process xlabel as it is an integer"""
            """Load image if path to image is provided"""
            if isinstance(image,str):
                image = imageio.imread(image)
            else:
                pass
            assert isinstance(image,np.ndarray)
            if isinstance(mask,str):
                mask = imageio.imread(mask)
            else:
                pass
            assert isinstance(mask,np.ndarray)
            """Validate the image and mask data and data structure"""
            assert len(image.shape) in (2,3)
            assert len(mask.shape) in (2,3)
            """Make the conventional 3-channel image and mask"""
            if len(image.shape)==2:
                image = np.expand_dims(image,axis=-1)
            else:
                assert image.shape[-1] in (1,3)
            if len(mask.shape)==2:
                mask = np.expand_dims(mask,-1)
            else:
                assert mask.shape[-1] == 1
            """Image color adjustment"""
            if image.shape[-1]==self.target_shape[-1]:
                pass
            else:
                if image.shape[-1]==1:
                    image = np.concatenate((image,image,image),axis=-1)
                else:
                    image = np.mean(image,axis=-1,keepdims=True).astype(image.dtype)
            """Image and mask size adjustment"""
            image = self.imresize(i_image=image, i_tsize=self.target_shape[0:2])
            mask  = self.scale_mask(i_mask=mask, i_tsize=self.target_shape[0:2])
            """Validate the final data"""
            assert isinstance(image,np.ndarray)
            assert isinstance(mask,np.ndarray)
            assert image.dtype in (np.uint8,)
            assert mask.dtype in (np.uint8,)
            """Accumulate data"""
            images.append(image)
            masks.append(mask)
            labels.append(label)
            count +=1
            """Write data to tfrecord file if the stacks were full"""
            if count == self.record_size:
                if segment == 0:
                    current_save_path = os.path.join(save_path, '{}.tfrecord'.format(save_file))
                else:
                    current_save_path = os.path.join(save_path, '{}_{}.tfrecord'.format(save_file, segment))
                if num_fields==3:
                    write_data = list(zip(images,masks,labels))
                else:
                    if 'mask' in db_fields.keys():
                        write_data = list(zip(images, masks))
                    else:
                        write_data = list(zip(images, labels))
                tfwriter.write(i_n_records=write_data, i_size=self.record_size, i_fields=db_fields,i_save_file=current_save_path)
                """Clear images and labels lists"""
                count = 0
                images.clear()
                masks.clear()
                labels.clear()
                segment += 1
            else:
                pass
        """Write the final part"""
        if len(images) > 0:
            if segment == 0:
                current_save_path = os.path.join(save_path, '{}.tfrecord'.format(save_file))
            else:
                current_save_path = os.path.join(save_path, '{}_{}.tfrecord'.format(save_file, segment))
            if num_fields == 3:
                write_data = list(zip(images, masks, labels))
            else:
                if 'mask' in db_fields.keys():
                    write_data = list(zip(images, masks))
                else:
                    write_data = list(zip(images, labels))
            tfwriter.write(i_n_records=write_data, i_size=self.record_size, i_fields=db_fields,i_save_file=current_save_path)
        else:
            pass
        dataset = tfwriter.read(i_tfrecord_path=i_save_path, i_original=True)
        return dataset
    """The main function"""
    def prepare(self,i_db=None,i_train_flag=True,i_lsm_factor=0.0,i_num_classes=1,i_pipeline_fn=None,**kwargs):
        """i_pipeline_fn is the pipeline function to parse the record obtained form tfrecord files"""
        """i_pipeline_fn(i_record=None,i_ori_shape:tuple=(256,256,1),i_train_flag:bool=True)"""
        assert isinstance(i_train_flag,bool)
        assert isinstance(i_lsm_factor,float)
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        save_path = os.path.join(self.save_path,'tfrecords')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        """Decide the base-name of tfrecord"""
        if i_train_flag:
            save_path = os.path.join(save_path,'train_db.tfrecord')
        else:
            save_path = os.path.join(save_path,'val_db.tfrecord')
        """Load dataset if it already saved to tfrecord"""
        if os.path.exists(save_path):
            dataset = TFRecordDB.read(i_tfrecord_path=save_path,i_original=True)  # Set i_original to True to return dictionary
        else:
            dataset = self.save_tfrecord(i_db=i_db, i_save_path=save_path)
        if 'only_save' in kwargs.keys():#Set this flag to only save dataset to tfrecord files.
            only_save = kwargs['only_save']
        else:
            only_save = False
        if only_save:
            return True
        else:
            pass
        dataset = dataset.map(lambda x: i_pipeline_fn(i_record=x, i_ori_shape=self.target_shape,i_lsm_factor=i_lsm_factor,i_num_classes=i_num_classes,i_train_flag=i_train_flag,**kwargs))
        return dataset
    """Support functions"""
    @classmethod
    def get_sample_db(cls, i_tsize=(224, 224), i_num_train_samples=None, i_num_val_samples=None):
        assert isinstance(i_tsize, (list, tuple)), 'Got type: {}'.format(type(i_tsize))
        assert len(i_tsize) == 2, 'Got lenght: {}'.format(len(i_tsize))
        assert i_tsize[0] > 0, 'Got value: {}'.format(i_tsize[0])
        assert i_tsize[1] > 0, 'Got value: {}'.format(i_tsize[1])
        def make_mask(i_image=None):
            assert isinstance(i_image, np.ndarray), 'Got type: {}'.format(type(i_image))
            assert i_image.dtype == np.uint8, 'Got type: {}'.format(i_image.dtype)
            image = (i_image > 127).astype(np.uint8)
            return image
        (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()
        if i_num_train_samples is None:
            i_num_train_samples = train_images.shape[0]
        else:
            assert isinstance(i_num_train_samples, int)
            assert i_num_train_samples > 0
        if i_num_val_samples is None:
            i_num_val_samples = val_images.shape[0]
        else:
            assert isinstance(i_num_val_samples, int)
            assert i_num_val_samples > 0
        train_images = [cls.imresize(i_image=image, i_tsize=i_tsize) for index, image in enumerate(train_images) if
                        index <= i_num_train_samples]
        train_labels = [label for index, label in enumerate(train_labels) if index <= i_num_train_samples]
        val_images   = [cls.imresize(i_image=image, i_tsize=i_tsize) for index, image in enumerate(val_images) if
                      index <= i_num_val_samples]
        val_labels   = [label for index, label in enumerate(val_labels) if index <= i_num_val_samples]
        train_masks  = [make_mask(i_image=image) for image in train_images]
        val_masks    = [make_mask(i_image=image) for image in val_images]
        train_db     = (train_images, train_masks, train_labels)
        val_db       = (val_images, val_masks, val_labels)
        return train_db, val_db
class DDataset:#This class is used for detection network
    def __init__(self):
        pass
if __name__ == '__main__':
    print('This module is to prepare data for a segmentation network')
"""=================================================================================================================="""