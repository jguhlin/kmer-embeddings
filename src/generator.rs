use bio::alignment::pairwise::*;
use bio::scores::blosum62;
use crossbeam::channel::{bounded, Receiver};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use rand_distr::weighted_alias::WeightedAliasIndex;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use std::thread::{JoinHandle, self};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};



static DNA_ALPHABET: [u8; 5] = *b"ACGTN";
static AA_ALPHABET: [u8; 23] = *b"ARNDCQEGHILKMFPSTWYVUOX";

pub struct KmerGenerator {
    k: usize,
    dist: WeightedAliasIndex<u32>,
    scoring_dist: Uniform<u8>,
    substitution_dist: Uniform<usize>,
    threads: usize,
    buffer_size: usize,
    rx: Option<Arc<Receiver<(Vec<u8>, Vec<u8>, i32)>>>,
    handles: Vec<JoinHandle<()>>,
    seed: u64,
    alphabet: Vec<u8>,
    alphabet_len: usize,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy)]
pub enum Alphabet {
    DNA,
    AA,
}

pub fn convert_to_onehot(seq: &[u8], alphabet: &[u8], alphabet_len: usize) -> Vec<u8> {
    let mut onehot = vec![0; seq.len() * alphabet_len];
    for (i, c) in seq.iter().enumerate() {
        let idx = alphabet.iter().position(|&x| x == *c).unwrap();
        onehot[i * alphabet_len + idx] = 1;
    }
    onehot
}

impl KmerGenerator {
    pub fn new() -> Self {
        KmerGenerator {
            k: 21,
            dist: WeightedAliasIndex::new(vec![25, 25, 25, 25, 5]).unwrap(),
            scoring_dist: Uniform::new_inclusive(0, 21),
            substitution_dist: Uniform::new_inclusive(0, 21),
            threads: 1,
            buffer_size: 32 * 1024,
            rx: None,
            handles: Vec::with_capacity(1),
            seed: 42,
            alphabet: DNA_ALPHABET.to_vec(),
            alphabet_len: 5,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn set_dna(&mut self) {
        self.alphabet = DNA_ALPHABET.to_vec();
        self.dist = WeightedAliasIndex::new(vec![25, 25, 25, 25, 5]).unwrap();
        self.alphabet_len = 5;
    }

    pub fn set_aa(&mut self) {
        self.alphabet = AA_ALPHABET.to_vec();
        let mut weights: [u32; 23] = [25; 23];
        weights[22] = 5;

        self.dist = WeightedAliasIndex::new(weights.to_vec()).unwrap();
        self.alphabet_len = 23;
    }

    pub fn set_k(&mut self, k: usize) {
        self.k = k;
        self.scoring_dist = Uniform::new(0, k as u8);
        self.substitution_dist = Uniform::new(0, k);
    }

    pub fn set_scoring(
        &mut self,
        match_score: i32,
        mismatch_score: i32,
        gap_open_score: i32,
        gap_extend_score: i32,
    ) {
        let scoring = Scoring::from_scores(
            match_score,
            mismatch_score,
            gap_open_score,
            gap_extend_score,
        );
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads;
    }

    pub fn set_weights(&mut self, weights: Vec<u32>) {
        assert!(weights.len() == self.alphabet_len);
        self.dist = WeightedAliasIndex::new(weights).unwrap();
    }

    pub fn set_capacity(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
    }

    pub fn start(&mut self) -> Arc<Receiver<(Vec<u8>, Vec<u8>, i32)>> {
        let (tx, rx) = bounded(self.buffer_size);
        self.rx = Some(Arc::new(rx));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);

        for _threadnum in 0..self.threads {
            rng.long_jump();
            let tx = tx.clone();
            let k = self.k;

            // Previously -5 for gaps
            let dna_scoring = Scoring::from_scores(-3, -1, 1, -1);

            let substitution_dist = Uniform::new(0, self.k);
            let dist = self.dist.clone();
            let mut aligner =
                Aligner::with_capacity_and_scoring(self.k, self.k, dna_scoring.clone());
            let mut blosum_aligner = Aligner::with_capacity(self.k, self.k, -5, -1, &blosum62);

            let mut rng = rng.clone();
            
            let alphabet = self.alphabet.clone();
            let alphabet_len = self.alphabet_len;

            let handle = std::thread::spawn(move || {
                let mut score;
                let backoff = crossbeam::utils::Backoff::new();
                let mut iter;
                let shutdown = Arc::new(AtomicBool::new(false));

                let mut k1: Vec<u8> = vec![0; k];
                let mut k2;

                let arch = pulp::Arch::new();

                'main: loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break 'main;
                    }
                    
                    backoff.reset();

                    iter = 0;

                    k1 = k1.iter().map(|_| alphabet[dist.sample(&mut rng)]).collect();
                    k2 = k1.clone();

                    // Get the maximum score for this kmer (self vs self)
                    let max_score = match alphabet_len {
                        5 => aligner.local(&k1, &k2).score,
                        23 => blosum_aligner.local(&k1, &k2).score,
                        _ => panic!("Invalid alphabet length"),
                    };

                    // Pick a random target score
                    let target_score = rng.gen_range(0..max_score);

                    score = 0;

                    // Introduce an indel
                    if target_score - score > (0.3 * max_score as f32) as i32 {
                        k2.remove(substitution_dist.sample(&mut rng));
                        k2.push(alphabet[dist.sample(&mut rng)]);
                    }

                    let align_score = match alphabet_len {
                        5 => aligner.local(&k1, &k2).score,
                        23 => blosum_aligner.local(&k1, &k2).score,
                        _ => panic!("Invalid alphabet length"),
                    };

                    score = max_score - align_score;

                    // If score diff is big enough, create a new random kmer
                    if target_score - score > (0.80 * max_score as f32) as i32 {
                        for x in 0..k {
                            k2[x] = alphabet[dist.sample(&mut rng)];
                        }
                    }

                    while score != target_score {
                        k2[substitution_dist.sample(&mut rng)] = alphabet[dist.sample(&mut rng)];

                        let align_score = match alphabet_len {
                            5 => aligner.local(&k1, &k2).score,
                            23 => blosum_aligner.local(&k1, &k2).score,
                            _ => panic!("Invalid alphabet length"),
                        };

                        score = max_score - align_score;

                        iter += 1;

                        if iter > 128 {
                            break;
                        }

                        if score > target_score {
                            k1 = k2.clone();
                            score = 0;
                        }
                    }

                    let mut msg = arch.dispatch(|| {
                        (
                            convert_to_onehot(&k1, &alphabet, alphabet_len),
                            convert_to_onehot(&k2, &alphabet, alphabet_len),
                            score,
                        )
                    });

                    while let Err(e) = tx.try_send(msg) {
                        match e {
                            crossbeam::channel::TrySendError::Full(msg_e) => {
                                msg = msg_e;
                                backoff.snooze();

                                if backoff.is_completed() {
                                    backoff.reset();
                                }
                            }
                            crossbeam::channel::TrySendError::Disconnected(_) => {
                                break 'main;
                            }
                        };
                    }
                }
            });

            self.handles.push(handle);
        }
        Arc::clone(self.rx.as_ref().unwrap())
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.rx = None;
        for handle in self.handles.drain(..) {
            handle.join().unwrap();
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aa_max_score() {
        let mut scores = Vec::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let k = 9;
        let mut k1: Vec<u8> = vec![0; k];
        let mut k2;

        let mut weights: [u32; 23] = [25; 23];
        weights[22] = 5;

        let dist = WeightedAliasIndex::new(weights.to_vec()).unwrap();
        let mut blosum_aligner = Aligner::with_capacity(k, k, -5, -1, &blosum62);

        for _ in 0..100_000 {
            for x in 0..k {
                k1[x] = AA_ALPHABET[dist.sample(&mut rng)];
            }

            k2 = k1.clone();

            scores.push(blosum_aligner.local(&k1, &k2).score);
        }

        let max_score = scores.iter().max().unwrap();
        let min_score = scores.iter().min().unwrap();
        let avg_score = scores.iter().sum::<i32>() as f32 / scores.len() as f32;
        println!("AA Max score: {}", max_score);
        println!("AA Min score: {}", min_score);
        println!("AA Avg score: {}", avg_score);
    }

    #[test]
    fn test_nt_max_score() {
        let mut scores = Vec::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let k = 21;
        let mut k1: Vec<u8> = vec![0; k];
        let mut k2;

        let scoring = Scoring::from_scores(-5, -1, 1, -1);
        let dist = WeightedAliasIndex::new(vec![25, 25, 25, 25, 5]).unwrap();
        let mut aligner = Aligner::with_capacity_and_scoring(k, k, scoring.clone());

        for _ in 0..100_000 {
            for x in 0..k {
                k1[x] = DNA_ALPHABET[dist.sample(&mut rng)];
            }

            k2 = k1.clone();

            scores.push(aligner.local(&k1, &k2).score);
        }

        let max_score = scores.iter().max().unwrap();
        let min_score = scores.iter().min().unwrap();
        let avg_score = scores.iter().sum::<i32>() as f32 / scores.len() as f32;
        println!("NT Max score: {}", max_score);
        println!("NT Min score: {}", min_score);
        println!("NT Avg score: {}", avg_score);
    }
}
