use kmer_embeddings::generator::KmerGenerator;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

// Register a `fibonacci` function and benchmark it over multiple cases.
#[divan::bench(args = [4, 8, 16, 32, 64, 128])]
fn nt_generator(k: usize) {
    let mut generator = KmerGenerator::new();
        generator.set_dna();
        generator.set_k(k);
        generator.set_threads(8);
    
    let rx = generator.start();

    let mut count = 0;
    for _ in rx.recv() {
        count += 1;
        if count >= 16 * 1024 * 1024 {
            break;
        }
    }
}

// Register a `fibonacci` function and benchmark it over multiple cases.
#[divan::bench(args = [4, 8, 16, 32, 64, 128])]
fn aa_generator(k: usize) {
    let mut generator = KmerGenerator::new();
        generator.set_aa();
        generator.set_k(k);
        generator.set_threads(8);
    
    let rx = generator.start();

    let mut count = 0;
    for _ in rx.recv() {
        count += 1;
        if count >= 16 * 1024 * 1024 {
            break;
        }
    }    
}