use burn::backend::autodiff::grads::Gradients;
use burn::lr_scheduler;
use burn::lr_scheduler::LrScheduler;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::AutodiffModule,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
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

use rerun::{demo_util::grid, external::glam};
use serde::{Deserialize, Serialize};

// Define the model configuration
#[derive(Config)]
pub struct KmerEmbeddingModel {
    pub kmer_size: usize,
    pub embedding_size: usize,
}

#[derive(Module, Debug, Clone)]
pub struct PoincareDistance {
    pub l2_norm: L2Norm,
}

impl PoincareDistance {
    pub fn new() -> Self {
        Self {
            l2_norm: L2Norm::new(),
        }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 2> {
        let x_norm = self.l2_norm.forward(x.clone());
        let y_norm = self.l2_norm.forward(y.clone());

        let diff_norm = self.l2_norm.forward(x - y).powf_scalar(2.0);

        let num = diff_norm * 2.0;
        let num = num.add_scalar(1e-12);
        let ones = Tensor::<B, 3>::ones_like(&x_norm);
        let denom = (ones.clone() - x_norm.clone().powf_scalar(2.0))
            * (ones - y_norm.clone().powf_scalar(2.0));
        let denom = denom.add_scalar(1e-12);

        let distance = num / denom;

        let distance = distance.clamp(1e-12, f32::MAX);

        let distance = distance.squeeze(2);

        acosh(distance.mul_scalar(2.0).add_scalar(1.0))
    }
}

#[derive(Module, Debug, Clone)]
pub struct L2Norm {}

impl L2Norm {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend, const N: usize>(&self, x: Tensor<B, N>) -> Tensor<B, N> {
        // Because you can't take the sqrt of 0
        x.powf_scalar(2.0)
            .sum_dim(N - 1)
            .clamp(1e-12, f32::MAX)
            .sqrt()
    }
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareTaxonomyEmbeddingModel<B: Backend> {
    pub embedding_token: Embedding<B>,
    l2_norm: L2Norm,
    poincare_distance: PoincareDistance,
    // scaling_inner: Linear<B>,
    scaling_layer: Linear<B>,
    // layer_norm: LayerNorm<B>,
}

// Define functions for model initialization
impl PoincareTaxonomyEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> PoincareTaxonomyEmbeddingModel<B> {
        let initializer = burn::nn::Initializer::Uniform {
            min: -0.005,
            max: 0.005,
        };

        //let layer_norm = LayerNormConfig::new(self.embedding_size)
        //.with_epsilon(self.layer_norm_eps)
        //.init(device);

        let embedding_token = EmbeddingConfig::new(self.taxonomy_size, self.embedding_size)
            .with_initializer(initializer)
            .init(device);

        let scaling_layer = LinearConfig::new(1, 1).with_bias(false);

        PoincareTaxonomyEmbeddingModel {
            embedding_token,
            l2_norm: L2Norm::new(),
            poincare_distance: PoincareDistance::new(),
            scaling_layer: scaling_layer.init(device),
            // layer_norm,
        }
    }
}

impl<B: Backend> PoincareTaxonomyEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(
        &self,
        origins: Tensor<B, 2, Int>,
        branches: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Float> {
        // let dims = branches.dims(); // Should be 32, but let's make it dynamic

        // println!("{}", origins);
        // println!("{}", branches);

        let origins = self.embedding_token.forward(origins);
        let destinations = self.embedding_token.forward(branches);
        let origins = origins.expand(destinations.dims());

        // Calculate the Poincaré distance
        let distances = self.poincare_distance.forward(origins, destinations);
        // distances.mul_scalar(100.0)
        // let distances: Tensor<B, 3> = distances.unsqueeze_dims(&[-1]);
        distances
        // self.scaling_layer.forward(distances).squeeze(2)

        /*

        // Simple euclidian

        // let distance = (origins - destinations).sum_dim(2); // .powf_scalar(2.0).sum_dim(2).sqrt();
        // println!("{}", distance);
        let distance = self.l2_norm.forward(origins - destinations);
        let dims = distance.dims();
        distance.squeeze(2)  */
    }

    pub fn forward_regression(
        &self,
        origins: Tensor<B, 2, Int>,
        pairs: Tensor<B, 2, Int>,
        distances: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let predicted_distances = self.forward(origins, pairs);
        // log::debug!("Predicted distances: {}", predicted_distances);
        // log::debug!("Expected distances: {}", distances);

        let loss = (predicted_distances.clone() - distances.clone())
            .powf_scalar(2.0)
            .mean();

        RegressionOutput::new(loss, predicted_distances, distances)
    }
}