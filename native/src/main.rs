use rust_lander::Env;
use std::time::Instant;
fn main(){
    let steps=std::env::args().nth(1).and_then(|x|x.parse::<u64>().ok()).unwrap_or(10_000_000);
    let mut env=Env::new(1,1000);let start=Instant::now();let mut episodes=0;let mut checksum=0.0;
    for i in 0..steps {
        let o=env.step((i%4) as u32).unwrap();checksum+=o[8] as f64;
        if env.done(){episodes+=1;env=Env::new(episodes,1000);}
    }
    let seconds=start.elapsed().as_secs_f64();
    println!("{{\"environment\":\"RustLander-v0\",\"steps\":{steps},\"seconds\":{seconds},\"steps_per_second\":{},\"episodes\":{episodes},\"checksum\":{checksum}}}",steps as f64/seconds);
}
