use burn::backend::autodiff::grads::Gradients;
use burn::lr_scheduler;
use burn::lr_scheduler::LrScheduler;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::AutodiffModule,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    optim::{
        adaptor::OptimizerAdaptor, AdamConfig, AdamWConfig, GradientsParams, Optimizer, SgdConfig,
        SimpleOptimizer,
    },
    prelude::*,
    record::CompactRecorder,
    record::Record,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{CpuTemperature, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
    LearningRate,
};

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::{WeightDecay, WeightDecayConfig};
use burn::optim::momentum::{Momentum, MomentumConfig, MomentumState};

use nn::loss::MseLoss;
use nn::{Linear, LinearConfig};
use rerun::{demo_util::grid, external::glam};
use serde::{Deserialize, Serialize};