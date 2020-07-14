"""
HRNetV2 - High Resolution Net model implementation in TensorFlow 2

# Reference

- [Deep High-Resolution Representation Learning for Visual Recognition] (https://arxiv.org/pdf/1908.07919.pdf)

# Reference implementations

- [HRNet] https://github.com/HRNet/HRNet-Image-Classification
- [hrnet-tf] https://github.com/yuanyuanli85/tf-hrnet
"""

import tensorflow as tf

layers = tf.keras.layers


class ConvModule(tf.keras.Model):
    """
    Module that combines convolutional layer, norm layer, and activation
    Order of layers is currently set to conv, norm, act
    """
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride,
                 padding='same',
                 use_bias=False,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(
                     2.0, mode='fan_out'),
                 weight_decay=1e-4,
                 norm_cfg=None,
                 act_cfg=None,
                 name=None):
        super(ConvModule, self).__init__()
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            use_bias=use_bias,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=name + '_conv' if name else None)
        self.norm = None
        if norm_cfg and norm_cfg['type'] == 'BN':
            bn_axis = norm_cfg.get('axis', -1)
            eps = norm_cfg.get('eps', 1e-5)
            momentum = norm_cfg.get('momentum', 0.997)
            self.norm = layers.BatchNormalization(axis=bn_axis,
                                                  epsilon=eps,
                                                  momentum=momentum,
                                                  name=name + '_bn')
        self.act = None
        if act_cfg:
            self.act = layers.Activation(act_cfg['type'],
                                         name=name +
                                         '_{}'.format(act_cfg['type']))

    def call(self, x, training=False):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 channels,
                 norm_cfg,
                 act_cfg,
                 expansion=1,
                 weight_decay=1e-4,
                 name=None,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_mod1 = ConvModule(channels,
                                    3,
                                    stride=stride,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod2 = ConvModule(channels,
                                    3,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None,
                                    name=name)

        self.downsample = downsample

    def call(self, x, training=False):
        residual = x
        x = self.conv_mod1(x, training=training)
        x = self.conv_mod2(x, training=training)

        if self.downsample:
            residual = self.downsample(x, training=training)

        x = x + residual
        x = tf.nn.relu(x)
        return x


class Bottleneck(tf.keras.Model):
    def __init__(self,
                 channels,
                 norm_cfg,
                 act_cfg,
                 expansion=4,
                 weight_decay=1e-4,
                 stride=1,
                 downsample=None,
                 name=None):
        super(Bottleneck, self).__init__()
        self.conv_mod1 = ConvModule(channels,
                                    1,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod2 = ConvModule(channels,
                                    3,
                                    stride=stride,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod3 = ConvModule(channels * expansion,
                                    1,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None,
                                    name=name)

        self.downsample = downsample

    def call(self, x, training=False):
        residual = x
        x = self.conv_mod1(x, training=training)
        x = self.conv_mod2(x, training=training)
        x = self.conv_mod3(x, training=training)

        if self.downsample:
            residual = self.downsample(x, training=training)

        x = x + residual
        x = tf.nn.relu(x)
        return x


class HighResolutionModule(tf.keras.Model):
    def __init__(self, cfg, multiscale_output=False):
        super(HighResolutionModule, self).__init__()
        self.weight_decay = cfg.get('weight_decay', 1e-4)
        self.norm_cfg = cfg.get('norm_cfg', None)
        self.act_cfg = cfg.get('act_cfg', None)
        self.num_branches = cfg['num_branches']
        self.num_blocks = cfg['num_blocks']
        self.num_channels = cfg['num_channels']
        assert self.num_branches == len(self.num_blocks)
        assert self.num_branches == len(self.num_channels)
        self.multiscale_output = multiscale_output
        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()

    def _make_branch(self, branch_level):
        blocks = []
        for block_index in range(self.num_blocks[branch_level]):
            block_name = 'hrm_{}_{}_{}'.format(len(self.num_channels),
                                                branch_level,
                                                block_index)
            blocks.append(
                BasicBlock(self.num_channels[branch_level],
                           self.norm_cfg,
                           self.act_cfg,
                           self.weight_decay,
                           name=block_name))
        return blocks

    def _make_branches(self):
        branches = []
        for i in range(self.num_branches):
            branches.append(self._make_branch(i))
        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:  # upsample low to high
                    conv_mod = ConvModule(self.num_channels[i],
                                          1,
                                          stride=1,
                                          padding='same',
                                          use_bias=False,
                                          weight_decay=self.weight_decay,
                                          norm_cfg=self.norm_cfg,
                                          act_cfg=self.act_cfg,
                                          name='fuse_{}_{}_{}'.format(
                                              self.num_branches, j, i))
                    upsample = layers.UpSampling2D(size=(2**(j - i),
                                                         2**(j - i)),
                                                   interpolation='bilinear')
                    fuse_layer.append(tf.keras.Sequential([conv_mod, upsample]))
                elif j == i:
                    fuse_layer.append(None)
                else:  # downsample 3x3 stride 2
                    down_layers = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_mod = ConvModule(
                                self.num_channels[i],
                                3,
                                stride=2,
                                padding='same',
                                use_bias=False,
                                weight_decay=self.weight_decay,
                                norm_cfg=self.norm_cfg,
                                act_cfg=None,
                                name='fuse_{}_{}_{}'.format(self.num_branches, j, i))
                        else:
                            conv_mod = ConvModule(
                                self.num_channels[j],
                                3,
                                stride=2,
                                padding='same',
                                use_bias=False,
                                weight_decay=self.weight_decay,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg,
                                name='fuse_{}_{}_{}'.format(self.num_branches, j, i))
                        down_layers.append(conv_mod)
                    fuse_layer.append(tf.keras.Sequential(down_layers))
            fuse_layers.append(fuse_layer)
        return fuse_layers

    def call(self, x, training=False):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i], training=training)
        if self.num_branches == 1:
            return [x[0]]
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0]
            if i > 0:
                y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(tf.nn.relu(y))
        return x_fuse


class Stem(tf.keras.Model):
    def __init__(self, cfg):
        super(Stem, self).__init__()
        filters = cfg['channels']
        kernel_size = cfg['kernel_size']
        stride = cfg['stride']
        use_bias = cfg['use_bias']
        padding = cfg['padding']
        weight_decay = cfg['weight_decay']

        self.conv_mod1 = ConvModule(filters,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    use_bias=use_bias,
                                    weight_decay=weight_decay,
                                    norm_cfg=cfg.get('norm_cfg', None),
                                    act_cfg=cfg.get('act_cfg', None),
                                    name='stem_1')

        self.conv_mod2 = ConvModule(filters,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    use_bias=use_bias,
                                    weight_decay=weight_decay,
                                    norm_cfg=cfg.get('norm_cfg', None),
                                    act_cfg=cfg.get('act_cfg', None),
                                    name='stem_2')

    def call(self, x, training=False):
        x = self.conv_mod1(x, training=training)
        x = self.conv_mod2(x, training=training)
        return x

class Transition(tf.keras.Model):
    def __init__(self, cfg, prev_layer_branches, prev_layer_channels):
        super(Transition, self).__init__()
        wd = cfg['weight_decay']
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        self.num_branches = cfg['num_branches']
        curr_stage_channels = cfg['num_channels']
        print(prev_layer_channels, curr_stage_channels)
        self.transition_layers = []
        for i in range(self.num_branches):
            if i < prev_layer_branches:
                if prev_layer_channels[i] != curr_stage_channels[i]:
                    convmod = ConvModule(curr_stage_channels[i],
                                         3,
                                         1,
                                         padding='same',
                                         use_bias=False,
                                         weight_decay=wd,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg,
                                         name='transition_{}_{}'.format(
                                             len(curr_stage_channels) - 1,
                                             i + 1))
                    self.transition_layers.append(convmod)
                    print('transition from', prev_layer_channels[i], 'to', curr_stage_channels[i])
                else:
                    print('no transition')
                    self.transition_layers.append(None)  # pass input as is
            else:
                # this handles the new branch(es) in the current stage
                new_transitions = []
                for j in range(i + 1 - prev_layer_branches):
                    if j == i - prev_layer_branches:
                        channels = curr_stage_channels[i]
                    else:
                        channels = prev_layer_channels[-1]
                    convmod = ConvModule(channels,
                                         3,
                                         2,
                                         padding='same',
                                         use_bias=False,
                                         weight_decay=wd,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg,
                                         name='new_transition_{}_{}'.format(
                                             len(curr_stage_channels) - 1,
                                             i + 1))
                    print('extra transition from', prev_layer_channels[-1], 'to', channels)
                    new_transitions.append(convmod)
                self.transition_layers.append(tf.keras.Sequential(*new_transitions))

    def call(self, x, training=False):
        outputs = []
        for tl in self.transition_layers:
            if tl:
                outputs.append(tl(x, training=training))
            else:
                outputs.append(x)
        for o in outputs:
            print(o.shape)
        return outputs


class Front(tf.keras.Model):
    def __init__(self, cfg, expansion=4):
        super(Front, self).__init__()
        wd = cfg['weight_decay']
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        num_blocks = cfg['num_blocks'][0]
        channels_list = cfg['num_channels']
        channels = channels_list[0]
        downsample = ConvModule(channels * expansion,
                                1,
                                1,
                                padding='same',
                                use_bias=False,
                                weight_decay=wd,
                                norm_cfg=norm_cfg,
                                act_cfg=None,
                                name="front_downsample")
        # block = residual unit
        self.blocks = []
        self.blocks.append(
            Bottleneck(channels,
                       norm_cfg,
                       act_cfg,
                       name='bottleneck_1',
                       weight_decay=wd,
                       stride=1,
                       downsample=downsample))
        for i in range(1, num_blocks):
            self.blocks.append(
                Bottleneck(channels,
                           norm_cfg,
                           act_cfg,
                           name='bottleneck_{}'.format(i + 1),
                           weight_decay=wd,
                           stride=1,
                           downsample=None))

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

class BottleneckStage(tf.keras.Model):
    def __init__(self,
                 channels,
                 num_blocks,
                 expansion=4,
                 stride=1,
                 weight_decay=1e-4,
                 norm_cfg=None,
                 act_cfg=None):
        super(BottleneckStage, self).__init__()
        downsample = ConvModule(channels * expansion,
                                1,
                                1,
                                padding='same',
                                use_bias=False,
                                weight_decay=weight_decay,
                                norm_cfg=norm_cfg,
                                act_cfg=None,
                                name="cls_downsample")
        self.blocks = []
        self.blocks.append(
            Bottleneck(channels,
                       norm_cfg,
                       act_cfg,
                       name='bottleneck_1',
                       weight_decay=weight_decay,
                       stride=1,
                       downsample=downsample))
        for i in range(1, num_blocks):
            self.blocks.append(
                Bottleneck(channels,
                           norm_cfg,
                           act_cfg,
                           name='bottleneck_{}'.format(i + 1),
                           weight_decay=weight_decay,
                           stride=1,
                           downsample=None))

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

class Stage(tf.keras.Model):
    def __init__(self, cfg, multiscale_output=True):
        super(Stage, self).__init__()
        self.num_modules = cfg['num_modules']
        self.num_branches = cfg['num_branches']
        self.modules = []
        for i in range(self.num_modules):
            hr_module = HighResolutionModule(
                cfg, multiscale_output=(i == self.num_modules - 1))
            self.modules.append(hr_module)

    def call(self, x_list, training=False):
        out = x_list
        for module in self.modules:
            out = module(out, training=training)
            print(out.shape, 'sssss')
        return out


class ClsHead(tf.keras.Model):
    def __init__(self, cfg, expansion=4):
        super(ClsHead, self).__init__()
        channels = cfg['channels']
        weight_decay = cfg.get('weight_decay', 1e-4)
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        num_classes = cfg.get('num_classes', 1000)
        fc_channels = cfg.get('fc_channels', 2048)
        # C, 2C, 4C, 8C -> 128, 256, 512, 1024
        self.width_incr_layers = []
        for i in range(len(channels)):
            incr_layer = BottleneckStage(channels[i],
                                         1,
                                         stride=1,
                                         weight_decay=weight_decay,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg)
            self.width_incr_layers.append(incr_layer)
        # downsampling layers
        self.downsample_layers = []
        for i in range(1, len(channels)):
            downsample = ConvModule(channels[i] * expansion,
                                    3,
                                    2,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name='downsample_cls_{}'.format(i))
            self.downsample_layers.append(downsample)
        self.final_layer = ConvModule(fc_channels,
                                      1,
                                      1,
                                      padding='same',
                                      use_bias=False,
                                      weight_decay=weight_decay,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg,
                                      name='final_{}'.format(i))
        self.classifier = layers.Dense(num_classes)


    def call(self, x, training=False):
        pass  #TODO


class HighResolutionNet(tf.keras.Model):
    def __init__(self, model_cfg):
        super(HighResolutionNet, self).__init__()

        # stem
        self.stem = Stem(model_cfg['stem'])
        print('created Stem')
        # stages
        self.stages = []
        self.transitions = []
        for s in range(1, model_cfg['num_stages'] + 1):
            stage_cfg = model_cfg['stage{}'.format(s)]
            if s == 1:
                # bottleneck units
                self.stages.append(Front(stage_cfg))
                print('created Stage ', s)
            else:
                # basic units
                prev_stage_cfg = model_cfg['stage{}'.format(s - 1)]
                prev_layer_branches = prev_stage_cfg['num_branches']
                prev_layer_channels = [prev_stage_cfg['expansion'] * c for c in prev_stage_cfg['num_channels']]
                self.transitions.append(Transition(stage_cfg, prev_layer_branches, prev_layer_channels))
                print('created Transition from ', s-1, s)
                self.stages.append(Stage(stage_cfg))
                print('created Stage ', s)

        # classification head
        head_cfg = model_cfg['head']
        self.cls_head = ClsHead(head_cfg)
        print('created Classification head')


    def call(self, x, training=False):
        x = self.stem(x, training=training)
        front = self.stages[0]
        stage1_output = front(x, training=training)
        print('front output', stage1_output.shape)
        transition12 = self.transitions[0]
        stage2 = self.stages[1]
        stage1_transitions = transition12(stage1_output, training=training)
        stage2_outputs = stage2(stage1_transitions) # list


        pass  #TODO
        return x


def test_hrnet():
    # CONFIG
    model = dict(type='HighResolutionNet',
                 num_stages=4,
                 stem=dict(
                     channels=64,
                     kernel_size=3,
                     stride=2,
                     padding='same',
                     use_bias=False,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ),
                 stage1=dict(
                     num_modules=1,
                     num_branches=1,
                     num_blocks=(4, ),
                     num_channels=(64, ),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ),
                 stage2=dict(
                     num_modules=1,
                     num_branches=2,
                     num_blocks=(4, 4),
                     num_channels=(32, 64),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ),
                 stage3=dict(
                     num_modules=4,
                     num_branches=3,
                     num_blocks=(4, 4, 4),
                     num_channels=(32, 64, 128),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ),
                 stage4=dict(
                     num_modules=3,
                     num_branches=4,
                     num_blocks=(4, 4, 4, 4),
                     num_channels=(32, 64, 128, 256),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ),
                 head=dict(
                     channels=(32, 64, 128, 256),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-4,
                     ),
                     weight_decay=1e-5,
                 ))
    train_cfg = dict(weight_decay=1e-5, )
    dataset_type = 'imagenet'
    dataset_mean = ()
    dataset_std = ()
    data_root = '/data/imagenet'
    data = dict(
        imgs_per_gpu=128,
        train=dict(
            type=dataset_type,
            train=True,
            dataset_dir=data_root,
            tf_record_pattern='train-*',
            resize_dim=256,
            crop_dim=224,
            augment=True,
            mean=(),
            std=(),
        ),

        val=dict(
            type=dataset_type,
            train=False,
            dataset_dir=data_root,
            tf_record_pattern='val-*',
            resize_dim=256,
            crop_dim=224,
            augment=False,
            mean=(),
            std=(),
        ),
    )
    evaluation = dict(interval=1)
    # optimizer
    optimizer = dict(
        type='SGD',
        learning_rate=1e-2,
        momentum=0.9,
        nesterov=True,
    )
    # extra options related to optimizers
    optimizer_config = dict(amp_enabled=True, )
    # learning policy
    lr_config = dict(policy='step',
                     warmup='linear',
                     warmup_epochs=5,
                     warmup_ratio=1.0 / 3,
                     step=[30, 60, 90])


    checkpoint_config = dict(interval=1, outdir='checkpoints')
    log_config = dict(interval=50, )
    total_epochs = 100,
    log_level = 'INFO'
    work_dir = './work_dirs/hrnet_w32_cls'
    resume_from = None

    hrnet = HighResolutionNet(model)
    # pass dummy data
    x = tf.random.uniform([8, 224, 224, 3])
    out = hrnet(x)
    hrnet.summary()


if __name__ == "__main__":
    test_hrnet()
