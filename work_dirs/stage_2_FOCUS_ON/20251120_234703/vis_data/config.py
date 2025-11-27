SYSTEM = ''
accumulative_counts = 8
batch_size = 1
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
]
data_path = '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/SlideBench/SlideBench-VQA-TCGA_smaller.csv'
dataloader_num_workers = 64
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 1000
evaluation_images = '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/WSI_feat/TCGA-A7-A0CJ-01Z-00-DX2.csv'
evaluation_inputs = [
    'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.',
]
image_path_list = None
launcher = 'none'
llava_dataset = dict(
    data_path=
    '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/SlideBench/SlideBench-VQA-TCGA_smaller.csv',
    dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
    image_folder=
    '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/WSI_feat/',
    image_path_list=None,
    max_length=19600,
    pad_image_to_square=False,
    per_image_length=2048,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.LLaVADataset')
llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 2e-05
max_epochs = 2
max_length = 19600
max_norm = 1
model = dict(
    freeze_llm=False,
    llm=dict(
        pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    pretrained_pth=
    '/home/ai-11/.cache/huggingface/hub/models--General-Medical-AI--SlideChat_Weight/snapshots/Model_Weights/stage1_pth/zero_pp_rank_0_mp_rank_00_model_states.pt',
    train_stage='2',
    type='xtuner.model.LLaVAModel',
    use_focus=True)
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    accumulative_counts=8,
    clip_grad=dict(error_if_nonfinite=False, max_norm=1),
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=2e-05,
        type='torch.optim.AdamW',
        weight_decay=0),
    type='mmengine.optim.AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
per_image_length = 2048
pretrained_pth = '/home/ai-11/.cache/huggingface/hub/models--General-Medical-AI--SlideChat_Weight/snapshots/Model_Weights/stage1_pth/zero_pp_rank_0_mp_rank_00_model_states.pt'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.qwen_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
sample_type = 'wsi'
save_steps = 500
save_total_limit = 2
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=2, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_path=
        '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/SlideBench/SlideBench-VQA-TCGA_smaller.csv',
        dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
        image_folder=
        '/home/ai-11/Public/SlideChat Model/SlideChat/dataset/WSI_feat/',
        image_path_list=None,
        max_length=19600,
        pad_image_to_square=False,
        per_image_length=2048,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.LLaVADataset'),
    num_workers=64,
    pin_memory=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0
work_dir = './work_dirs/stage_2_FOCUS_ON'
