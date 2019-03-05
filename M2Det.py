# coding:utf-8
import keras
from keras.layers import *
from keras.initializers import *
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.models import Model


class M2Det:
    def __init__(self):
        self.batch_size = 8
        self.input_shape = self.batch_size,320,320,3
        self.backbone = 'resnet101'
        self.TUM_number = 8
        self.anchor_num = 6
        self.class_num = 5

    def model(self):
        input = Input(name='input',batch_shape=self.input_shape)
        print(input.shape,Flatten()(input).shape)
        f2,f1 = self.back_bone(input)
        #f1's channles is less than f2's
        base_feature = self.FFMv1(f1,f2)
        pre_feature = None
        pyramids = []
        for i in range(1,self.TUM_number + 1):
            if i == 1 :
                f = self.Conv(base_feature,filters=256,kernel_size=(1,1),strides=(1,1),
                          padding='valid',conv_name='conv_FFMv2_1',bn_name='bn_FFMv2_1')
            else:
                f = self.FFMv2(base_feature,pre_feature,i)
            out = self.TUM(f,i)
            pre_feature = out[0]
            if i == 1:
                pyramids = out
            else:
                for index in range(6):
                    pyramids[index] = Concatenate()([pyramids[index],out[index]])

        pyramids = self.SFAM(pyramids)
        predition = self.detection(pyramids)
        model = keras.Model(inputs=input, outputs=predition)
        return model

    def FFMv1(self,feature1,feature2):
        x1_channel = np.int32(feature1.shape[3])/ np.int32(2)
        x2_channel = np.int32(feature2.shape[3])/ np.int32(2)
        x1 = self.Conv(img=feature1,filters=np.int32(x1_channel),kernel_size=(1,1),
                       strides=(1,1),padding='valid',conv_name='conv_FFMv1_f1',
                       bn_name='bn_FFMv1_f1')
        x1 = UpSampling2D(size=(2,2))(x1)

        x2 = self.Conv(img=feature2, filters=np.int32(x2_channel), kernel_size=(3, 3),
                       strides=(1, 1), padding='same', conv_name='conv_FFMv1_f2',
                       bn_name='bn_FFMv1_f2')
        x = Concatenate()([x1,x2])
        return x

    def FFMv2(self,base_feture,pre_feature,stage):
        pre_channel = np.int32(pre_feature.shape[3])
        x = self.Conv(base_feture,filters=pre_channel,kernel_size=(1,1),
                      strides=(1,1),padding='valid',
                      conv_name='FFMv2_conv_'+str(stage),bn_name='FFMv2_bn_'+str(stage))

        o = Concatenate()([x,pre_feature])
        return o

    def Conv(self,img,filters,kernel_size,strides,padding,conv_name,bn_name):
        o = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,
                   padding=padding,name=conv_name)(img)
        o = BatchNormalization(axis=3,name=bn_name)(o)
        o = Activation('relu')(o)
        return o

    def TUM(self,x,stage):
        # TUM module
        conv_TUM_name_base = 'conv_' + str(stage) + '_level'
        bn_TUM_name_base = 'bn_' + str(stage) + '_level'
        channel_in = np.int32(x.shape[3])
        channel_out = np.int32(np.int32(x.shape[3]) / np.int32(2))
        size_buffer = []

        # level one
        level = 1
        f1 = x
        size_buffer.append([int(f1.shape[2])] * 2)

        f2 = self.Conv(f1, channel_in, (3, 3), (2, 2), 'same',
                       conv_TUM_name_base + str(level) + '_1',
                       bn_TUM_name_base + str(level) + '_1')
        size_buffer.append([int(f2.shape[2])] * 2)

        f3 = self.Conv(f2, channel_in, (3, 3), (2, 2), 'same',
                       conv_TUM_name_base + str(level) + '_2',
                       bn_TUM_name_base + str(level) + '_2')
        size_buffer.append([int(f3.shape[2])] * 2)

        f4 = self.Conv(f3, channel_in, (3, 3), (2, 2), 'same',
                       conv_TUM_name_base + str(level) + '_3',
                       bn_TUM_name_base + str(level) + '_3')
        size_buffer.append([int(f4.shape[2])] * 2)

        f5 = self.Conv(f4, channel_in, (3, 3), (2, 2), 'same',
                       conv_TUM_name_base + str(level) + '_4',
                       bn_TUM_name_base + str(level) + '_4')
        size_buffer.append([int(f5.shape[2])] * 2)

        f6 = self.Conv(f5, channel_in, (3, 3), (2, 2), 'valid',
                       conv_TUM_name_base + str(level) + '_5',
                       bn_TUM_name_base + str(level) + '_5')
        size_buffer.append([int(f6.shape[2])] * 2)

        # level two:using Blinear Upsample + ele-wise sum
        # define a Lambda function to compute upsample_blinear
        level = 2
        c6 = f6

        c5 = self.Conv(c6, channel_in, (3, 3), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_5',
                       bn_TUM_name_base + str(level) + '_5')
        c5 = Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[4]))(c5)
        c5 = Add()([c5, f5])

        c4 = self.Conv(c5, channel_in, (3, 3), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_4',
                       bn_TUM_name_base + str(level) + '_4')
        c4 = Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[3]))(c4)
        c4 = Add()([c4, f4])

        c3 = self.Conv(c4, channel_in, (3, 3), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_3',
                       bn_TUM_name_base + str(level) + '_3')
        c3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[2]))(c3)
        c3 = Add()([c3, f3])

        c2 = self.Conv(c3, channel_in, (3, 3), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_2',
                       bn_TUM_name_base + str(level) + '_2')
        c2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[1]))(c2)
        c2 = Add()([c2, f2])

        c1 = self.Conv(c2, channel_in, (3, 3), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_1',
                       bn_TUM_name_base + str(level) + '_1')
        c1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[0]))(c1)
        c1 = Add()([c1, f1])

        # level three:using 1 * 1 kernel to make it smooth
        level = 3
        o1 = Add()([c1, f1])
        o1 = self.Conv(o1, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_1',
                       bn_TUM_name_base + str(level) + '_1')

        o2 = Add()([c2, f2])
        o2 = self.Conv(o2, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_',
                       bn_TUM_name_base + str(level) + '_2')

        o3 = Add()([c3, f3])
        o3 = self.Conv(o3, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_3',
                       bn_TUM_name_base + str(level) + '_3')

        o4 = Add()([c4, f4])
        o4 = self.Conv(o4, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_4',
                       bn_TUM_name_base + str(level) + '_4')

        o5 = Add()([c5, f5])
        o5 = self.Conv(o5, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_5',
                       bn_TUM_name_base + str(level) + '_5')

        o6 = Add()([c6, f6])
        o6 = self.Conv(o6, channel_out, (1, 1), (1, 1), 'same',
                       conv_TUM_name_base + str(level) + '_6',
                       bn_TUM_name_base + str(level) + '_6')
        # print(o1.shape,o2.shape,o3.shape,o4.shape,o5.shape,o6.shape)
        return [o1, o2, o3, o4, o5, o6]



    def SFAM(self,pyramids):
        '''
        :param pyramids: list of features
        :return:
        '''
        for p in pyramids:
            # squeeze stage: global average pooling
            s = GlobalAvgPool2D(data_format='channels_last',name='global_ave_pool_SFAM')(p)
            # excitation stage: use two Fully-connection layers
            fc_1 = Dense(64,activation='sigmoid',use_bias=False,
                         kernel_initializer=glorot_uniform(seed=0))(s)
            fc_2 = Dense(1024,activation='sigmoid',use_bias=False,
                         kernel_initializer=glorot_uniform(seed=0))(fc_1)
            #reweight
            p = Multiply()([p,fc_2])

        return pyramids

    def res_block(self,input,number,stride,stage,kernel_size,part):

        con2d_name_base = 'conv' + str(stage) + '_' +str(part)
        bn_name_base = 'bn' + str(stage) + '_' +str(part)
        n1, n2, n3 = number

        SEED = 0
        short_cut = input

        x = Conv2D(filters=n1,strides=stride,padding='same',
                   kernel_size=(1,1),kernel_initializer=glorot_normal(seed=SEED),name=con2d_name_base+'_a')(input)
        x = BatchNormalization(axis=3,name=bn_name_base + '_a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=n2, strides=stride, padding='same', kernel_size=kernel_size,
                   kernel_initializer=glorot_normal(seed=SEED), name=con2d_name_base + '_b')(x)
        x = BatchNormalization(axis=3,name = bn_name_base + '_b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=n3, strides=stride, padding='same', kernel_size=(1, 1),
                   kernel_initializer=glorot_normal(seed=SEED), name=con2d_name_base + '_c')(x)
        x = BatchNormalization(axis=3,name = bn_name_base + '_c')(x)

        out = Add()([x,short_cut])
        out = Activation('relu')(out)
        return out

    def res_block_first(self,input,number,stride,stage,kernel_size,part):

        con2d_name_base = 'conv' + str(stage) + '_' +str(part)
        bn_name_base = 'bn' + str(stage) + '_' +str(part)

        n1,n2,n3 = number
        SEED = 0

        short_cut = Conv2D(filters=n3,strides=stride,name=con2d_name_base+'_skip',kernel_size=(1,1))(input)
        short_cut = BatchNormalization(axis=3,name=bn_name_base+'_skip')(short_cut)


        x = Conv2D(filters=n1,strides=stride,padding='same',
                   kernel_size=(1,1),kernel_initializer=glorot_normal(seed=SEED),name=con2d_name_base+'_a')(input)
        x = BatchNormalization(axis=3,name=bn_name_base + '_a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=n2, strides=(1,1), padding='same', kernel_size=kernel_size,
                   kernel_initializer=glorot_normal(seed=SEED), name=con2d_name_base + '_b')(x)
        x = BatchNormalization(axis=3,name = bn_name_base + '_b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=n3, strides=(1,1), padding='same', kernel_size=(1, 1),
                   kernel_initializer=glorot_normal(seed=SEED), name=con2d_name_base + '_c')(x)
        x = BatchNormalization(axis=3,name = bn_name_base + '_c')(x)

        out = Add()([x,short_cut])
        out = Activation('relu')(out)
        return out

    def back_bone(self,input):
        # stage1
        x = ZeroPadding2D((3,3),name='stage1_zero_pading')(input)
        x = Conv2D(64,kernel_size=(7,7),name = 'conv_stage1',strides=(2,2),kernel_initializer = glorot_uniform(0))(x)
        x = BatchNormalization(axis=3,name = 'bn_satge1')(x)
        x = Activation('relu',name = 'relu_stage1')(x)
        x = MaxPool2D(pool_size=(3,3),strides=(2,2),name = 'pool_stage1')(x)

        # stage2
        stage = 2
        x = self.res_block_first(input= x,number = [64,64,256],stride = (1,1),kernel_size=(3,3),stage = stage,part = 1)
        x = self.res_block(input = x,number = [64,64,256],stride = (1, 1),kernel_size=(3,3),stage = stage,part = 2)
        x = self.res_block(input = x,number = [64,64,256],stride = (1, 1),kernel_size=(3,3),stage = stage,part = 3)

        # stage3
        stage = 3
        x = self.res_block_first(input=x, number=[128, 128, 512], stride=(2, 2), kernel_size = (3,3), stage = stage,part=1)
        x = self.res_block(input = x,number = [128, 128, 512],stride = (1, 1),kernel_size=(3,3),stage = stage,part=2)
        x = self.res_block(input = x,number = [128, 128, 512],stride = (1, 1),kernel_size=(3,3),stage = stage,part=3)
        x = self.res_block(input = x,number = [128, 128, 512],stride = (1, 1),kernel_size=(3,3),stage = stage,part=4)
        #Get feature1 shape:512*40*40
        feature1 = x
        # stage4
        stage = 4
        x = self.res_block_first(input=x, number=[256, 256, 1024], stride=(2, 2), kernel_size=(3, 3), stage=stage,part=1)
        for i in range(2,24):
            x = self.res_block(input=x, number=[256, 256, 1024], stride=(1, 1), kernel_size=(3, 3), stage=stage,part=i)
        ##Get feature1 shape:1024*20*20
        feature2 = x
        # stage5
        # stage = 5
        # x = self.res_block_first(input=x, number=[512,512,2048], stride=(2, 2), kernel_size=(3, 3), stage=stage,part=1)
        # x = self.res_block(input=x, number=[512,512,2048], stride=(1, 1), kernel_size=(3, 3), stage=stage,part=2)
        # x = self.res_block(input=x, number=[512,512,2048], stride=(1, 1), kernel_size=(3, 3), stage=stage,part=3)
        return feature1,feature2

    def detection(self,pyramids):
        all_cls = []
        all_reg = []
        for p in pyramids:
            print(p.shape)
            cls = Conv2D(filters=self.anchor_num * self.class_num,kernel_size=(3,3),
                         strides=(1,1),padding = 'same')(p)
            cls = Flatten()(cls)
            all_cls.append(cls)

            reg = Conv2D(filters=self.anchor_num * 4, kernel_size=(3, 3),
                         strides=(1, 1), padding='same')(p)
            reg = Flatten()(reg)
            all_reg.append(reg)

        all_cls = Concatenate()(all_cls)
        all_reg = Concatenate()(all_reg)
        print(all_reg.shape,type(all_reg.shape[-1]))
        box_num = int(int(all_reg.shape[-1]) / 4)
        all_cls = Reshape((-1,box_num,self.class_num))(all_cls)
        all_cls = Softmax()(all_cls)
        all_reg = Reshape((-1,box_num,4))(all_reg)

        prediction = Concatenate()([all_reg,all_cls])
        return prediction

if __name__ == '__main__':
    model = M2Det().model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, 'M2Det.png', show_shapes=True)





