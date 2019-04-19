
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'gc_gan_share':
        assert(opt.dataset_mode == 'unaligned')
        from .gc_gan_share_model import GcGANShareModel
        model = GcGANShareModel()
    elif opt.model == 'gc_gan_mix':
        assert(opt.dataset_mode == 'unaligned')
        from .gc_gan_mix_model import GcGANMixModel
        model = GcGANMixModel()
    elif opt.model == 'gc_gan_mix_random':
        assert(opt.dataset_mode == 'unaligned')
        from .gc_gan_mix_random_model import GcGANMixRandomModel
        model = GcGANMixRandomModel()
    elif opt.model == 'gc_gan_cross':
        assert(opt.dataset_mode == 'unaligned')
        from .gc_gan_cross_model import GcGANCrossModel
        model = GcGANCrossModel()
    elif opt.model == 'gc_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .gc_cycle_gan_model import GcCycleGANModel
        model = GcCycleGANModel()
    elif opt.model == 'gan_alone':
        assert(opt.dataset_mode == 'unaligned')
        from .gan_alone_model import GANAloneModel
        model = GANAloneModel()
    elif opt.model == 'test_gcgan':
        assert(opt.dataset_mode == 'unaligned')
        from .test_gcgan_model import TestGcGANModel
        model = TestGcGANModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
