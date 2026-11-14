//! RustLander-v0: deliberately simplified landing task, NOT Gymnasium Box2D.
use std::ffi::c_void;
pub const OBS: usize = 8;
pub const OUT: usize = 12;
const DT: f32 = 0.02;

#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    fn new(seed:u64)->Self {Self(seed)}
    fn unit(&mut self)->f32 {
        self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);
        ((z^(z>>31))>>40) as f32 / 16777216.0
    }
    fn centered(&mut self,scale:f32)->f32 {(self.unit()*2.0-1.0)*scale}
}
#[derive(Clone)]
pub struct Env {
    pub state:[f32;OBS],ticks:u32,horizon:u32,shaping:f32,
    pub terminated:bool,pub truncated:bool,pub success:bool,
}
impl Env {
    pub fn new(seed:u64,horizon:u32)->Self {
        let mut rng=Rng::new(seed);
        let state=[rng.centered(0.25),1.2,rng.centered(0.15),rng.centered(0.1),rng.centered(0.1),rng.centered(0.1),0.0,0.0];
        let mut e=Self{state,ticks:0,horizon:horizon.max(1),shaping:0.0,terminated:false,truncated:false,success:false};
        e.shaping=e.potential();e
    }
    pub fn done(&self)->bool {self.terminated||self.truncated}
    fn potential(&self)->f32 {
        let s=self.state;
        -100.0*s[0].hypot(s[1])-100.0*s[2].hypot(s[3])-100.0*s[4].abs()+10.0*(s[6]+s[7])
    }
    pub fn output(&self,reward:f32)->[f32;OUT] {
        let mut o=[0.0;OUT];o[..OBS].copy_from_slice(&self.state);
        o[8]=reward;o[9]=self.terminated as u8 as f32;o[10]=self.truncated as u8 as f32;o[11]=self.success as u8 as f32;o
    }
    pub fn step(&mut self,action:u32)->Result<[f32;OUT], &'static str> {
        if action>3 {return Err("invalid action");}
        if self.done(){return Err("reset required");}
        self.ticks+=1;
        let main=if action==2 {3.2}else{0.0};
        let side=match action {1=>1.0,3=>-1.0,_=>0.0};
        let s=&mut self.state;
        // Positive angle rotates the upward thrust leftward. Side engines alter torque.
        s[2]=(s[2]+(-main*s[4].sin()-0.08*side)*DT)*0.999;
        s[3]=(s[3]+(main*s[4].cos()-1.5)*DT)*0.999;
        s[5]=(s[5]+4.0*side*DT)*0.98;
        s[4]+=s[5]*DT;s[0]+=s[2]*DT;s[1]+=s[3]*DT;
        let touchdown=s[1]<=0.0;
        if touchdown {
            s[1]=0.0;
            self.success=s[0].abs()<=0.20 && s[2].abs()<=0.25 && s[3].abs()<=0.30 && s[4].abs()<=0.20 && s[5].abs()<=0.40;
            if self.success {s[6]=1.0;s[7]=1.0;}
        }
        self.terminated=touchdown||s[0].abs()>1.0||s[1]>2.5||s[4].abs()>std::f32::consts::PI;
        self.truncated=!self.terminated && self.ticks>=self.horizon;
        let new_shaping=self.potential();
        let mut reward=new_shaping-self.shaping-if action==2 {0.3}else if action==0 {0.0}else{0.03};
        self.shaping=new_shaping;
        if self.terminated {reward=if self.success{100.0}else{-100.0};}
        Ok(self.output(reward))
    }
}
struct Batch {envs:Vec<Env>}
#[no_mangle] pub extern "C" fn rl_version()->u32 {1}
#[no_mangle] pub extern "C" fn rl_create(count:u32,horizon:u32)->*mut c_void {
    if count==0||count>65536||horizon==0{return std::ptr::null_mut();}
    Box::into_raw(Box::new(Batch{envs:(0..count).map(|i|Env::new(i as u64,horizon)).collect()})) as *mut c_void
}
#[no_mangle] pub unsafe extern "C" fn rl_destroy(h:*mut c_void) {if !h.is_null(){drop(Box::from_raw(h as *mut Batch));}}
#[no_mangle] pub unsafe extern "C" fn rl_reset(h:*mut c_void,index:u32,seed:u64,out:*mut f32)->i32 {
    if h.is_null()||out.is_null(){return -1;}
    let b=&mut*(h as *mut Batch);let Some(e)=b.envs.get_mut(index as usize)else{return -2;};
    *e=Env::new(seed,e.horizon);std::ptr::copy_nonoverlapping(e.output(0.0).as_ptr(),out,OUT);0
}
#[no_mangle] pub unsafe extern "C" fn rl_step(h:*mut c_void,actions:*const u32,out:*mut f32)->i32 {
    if h.is_null()||actions.is_null()||out.is_null(){return -1;}
    let b=&mut*(h as *mut Batch);let a=std::slice::from_raw_parts(actions,b.envs.len());
    if a.iter().any(|x|*x>3){return -2;}
    if b.envs.iter().any(Env::done){return -3;}
    let o=std::slice::from_raw_parts_mut(out,b.envs.len()*OUT);
    for ((e,a),o) in b.envs.iter_mut().zip(a).zip(o.chunks_mut(OUT)) {o.copy_from_slice(&e.step(*a).unwrap());}0
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn replay_and_seed_variation(){
        let mut a=Env::new(42,1000);let mut b=a.clone();assert_ne!(a.state,Env::new(43,1000).state);
        for i in 0..1000 {assert_eq!(a.step(i%4),b.step(i%4));if a.done(){break;}}
    }
    #[test] fn invalid_action_is_atomic(){let mut a=Env::new(42,100);let before=a.output(0.0);assert!(a.step(4).is_err());assert_eq!(before,a.output(0.0));}
    #[test] fn time_limit_is_not_landing(){let mut a=Env::new(42,1);a.step(0).unwrap();assert!(a.truncated);assert!(!a.terminated);assert!(!a.success);assert!(a.step(0).is_err());}
    #[test] fn soft_landing_and_crash(){
        let mut a=Env::new(1,100);a.state=[0.0,0.001,0.0,-0.1,0.0,0.0,0.0,0.0];
        let o=a.step(0).unwrap();assert!(a.success&&a.terminated&&!a.truncated);assert_eq!(o[8],100.0);
        let mut b=Env::new(1,100);b.state=[0.0,0.001,0.0,-1.0,0.0,0.0,0.0,0.0];
        assert_eq!(b.step(0).unwrap()[8],-100.0);assert!(b.terminated&&!b.success);
    }
    #[test] fn thrust_and_rotation(){
        let a=Env::new(123,100);let mut idle=a.clone();let mut thrust=a.clone();let mut left=a.clone();let mut right=a;
        idle.step(0).unwrap();thrust.step(2).unwrap();left.step(1).unwrap();right.step(3).unwrap();
        assert!(thrust.state[3]>idle.state[3]);assert!(left.state[5]>right.state[5]);
    }
    #[test] fn finite_and_lifecycle(){for seed in 0..100 {let mut a=Env::new(seed,1000);let mut r=Rng::new(seed);while !a.done(){let o=a.step((r.unit()*4.0) as u32).unwrap();assert!(o.iter().all(|x|x.is_finite()));}}}
}
