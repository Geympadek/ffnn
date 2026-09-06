use crate::params::{Params};
use std::iter::{Chain, Once, Empty};
use std::slice::{Chunks, ChunksMut, Iter, IterMut};

pub trait Layer {
    fn new() -> Self;

    type BiasesIter<'a>: Iterator<Item=&'a [f32]> where Self: 'a;
    type BiasesIterMut<'a>: Iterator<Item=&'a mut [f32]> where Self: 'a;
    
    type WeightsIter<'a>: Iterator<Item=std::slice::Chunks<'a, f32>> where Self: 'a;
    type WeightsIterMut<'a>: Iterator<Item=std::slice::ChunksMut<'a, f32>> where Self: 'a;
    
    type BiasesBuff<'a>: Iterator<Item=&'a f32> where Self: 'a;
    type BiasesBuffMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    type WeightsBuff<'a>: Iterator<Item=&'a f32> where Self: 'a;
    type WeightsBuffMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    fn biases_iter(&self) -> Self::BiasesIter<'_>;
    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_>;
    
    fn weights_iter(&self) -> Self::WeightsIter<'_>;
    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_>;
    
    fn biases_buff(&self) -> Self::BiasesBuff<'_>;
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_>;

    fn weights_buff(&self) -> Self::WeightsBuff<'_>;
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_>;
}

pub struct WorkingLayer<
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    Tail,
>{
    weights: [[f32; INPUT_LEN]; OUTPUT_LEN],
    biases: [f32; OUTPUT_LEN],
    next: Tail
}

impl<
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    Tail: Layer
> Layer for WorkingLayer<INPUT_LEN, OUTPUT_LEN, Tail> {
    type BiasesIter<'a> = Chain<Once<&'a [f32]>, Tail::BiasesIter<'a>> where Self: 'a;
    type BiasesIterMut<'a> = Chain<Once<&'a mut [f32]>, Tail::BiasesIterMut<'a>> where Self: 'a;
    
    type WeightsIter<'a> = Chain<Once<Chunks<'a, f32>>, Tail::WeightsIter<'a>> where Self: 'a;
    type WeightsIterMut<'a> = Chain<Once<ChunksMut<'a, f32>>, Tail::WeightsIterMut<'a>> where Self: 'a;
    
    type BiasesBuff<'a> = Chain<Iter<'a, f32>, Tail::BiasesBuff<'a>> where Self: 'a;
    type BiasesBuffMut<'a> = Chain<IterMut<'a, f32>, Tail::BiasesBuffMut<'a>> where Self: 'a;

    type WeightsBuff<'a> = Chain<Iter<'a, f32>, Tail::WeightsBuff<'a>> where Self: 'a;
    type WeightsBuffMut<'a> = Chain<IterMut<'a, f32>, Tail::WeightsBuffMut<'a>> where Self: 'a;

    fn new() -> Self {
        let weights = [[0.0f32; INPUT_LEN]; OUTPUT_LEN];
        let biases = [0.0f32; OUTPUT_LEN];
        Self{
            weights,
            biases,
            next: Tail::new()
        }
    }

    fn biases_iter(&self) -> Self::BiasesIter<'_> {
        std::iter::once(&self.biases[..]).chain(self.next.biases_iter())
    }
    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_> {
        std::iter::once(&mut self.biases[..]).chain(self.next.biases_iter_mut())
    }

    fn weights_iter(&self) -> Self::WeightsIter<'_> {
        std::iter::once(self.weights.as_flattened().chunks(INPUT_LEN)).chain(self.next.weights_iter())
    }
    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_> {
        std::iter::once(self.weights.as_flattened_mut().chunks_mut(INPUT_LEN)).chain(self.next.weights_iter_mut())
    }

    fn biases_buff(&self) -> Self::BiasesBuff<'_> {
        self.biases.iter().chain(self.next.biases_buff())
    }
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_> {
        self.biases.iter_mut().chain(self.next.biases_buff_mut())
    }

    fn weights_buff(&self) -> Self::WeightsBuff<'_> {
        self.weights.as_flattened().iter().chain(self.next.weights_buff())
    }
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_> {
        self.weights.as_flattened_mut().iter_mut().chain(self.next.weights_buff_mut())
    }
}

pub struct OutputLayer<const INPUT_LEN: usize> {}

impl<const INPUT_LEN: usize> Layer for OutputLayer<INPUT_LEN> {
    fn new() -> Self {
        Self {}
    }

    type BiasesIter<'a> = Empty<&'a [f32]>;
    type BiasesIterMut<'a> = Empty<&'a mut [f32]>;

    type WeightsIter<'a> = Empty<Chunks<'a, f32>>;
    type WeightsIterMut<'a> = Empty<ChunksMut<'a, f32>>;

    type BiasesBuff<'a> = Empty<&'a f32>;
    type BiasesBuffMut<'a> = Empty<&'a mut f32>;

    type WeightsBuff<'a> = Empty<&'a f32>;
    type WeightsBuffMut<'a> = Empty<&'a mut f32>;

    fn biases_iter(&self) -> Self::BiasesIter<'_> {
        std::iter::empty()
    }

    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_> {
        std::iter::empty()
    }

    fn weights_iter(&self) -> Self::WeightsIter<'_> {
        std::iter::empty()
    }
    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_> {
        std::iter::empty()
    }

    fn biases_buff(&self) -> Self::BiasesBuff<'_> {
        std::iter::empty()
    }
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_> {
        std::iter::empty()
    }

    fn weights_buff(&self) -> Self::WeightsBuff<'_> {
        std::iter::empty()
    }
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_> {
        std::iter::empty()
    }
}

pub struct ParamsStack<Layers> {
    layers: Layers
}

impl<Layers: Layer> ParamsStack<Layers> {
    pub fn new() -> Self {
        Self {
            layers: Layers::new()
        }
    }
}

impl<Layers: Layer> Params for ParamsStack<Layers> {
    type BiasesIter<'a> = Layers::BiasesIter<'a> where Self: 'a;
    type BiasesIterMut<'a> = Layers::BiasesIterMut<'a> where Self: 'a;

    type WeightsIter<'a> = Layers::WeightsIter<'a> where Self: 'a;
    type WeightsIterMut<'a> = Layers::WeightsIterMut<'a> where Self: 'a;

    type WeightsRowIter<'a> = Chunks<'a, f32> where Self: 'a;
    type WeightsRowIterMut<'a> = ChunksMut<'a, f32> where Self: 'a;
    
    type BiasesBuff<'a> = Layers::BiasesBuff<'a> where Self: 'a;
    type BiasesBuffMut<'a> = Layers::BiasesBuffMut<'a> where Self: 'a;

    type WeightsBuff<'a> = Layers::WeightsBuff<'a> where Self: 'a;
    type WeightsBuffMut<'a> = Layers::WeightsBuffMut<'a> where Self: 'a;

    type RawParamIter<'a> = Chain<Self::BiasesBuff<'a>, Self::WeightsBuff<'a>> where Self: 'a;
    type RawParamIterMut<'a> = Chain<Self::BiasesBuffMut<'a>, Self::WeightsBuffMut<'a>> where Self: 'a;

    fn biases_iter(&self) -> Self::BiasesIter<'_> {
        self.layers.biases_iter()
    }
    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_> {
        self.layers.biases_iter_mut()
    }
    fn weights_iter(&self) -> Self::WeightsIter<'_> {
        self.layers.weights_iter()
    }
    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_> {
        self.layers.weights_iter_mut()
    }

    fn biases_buff(&self) -> Self::BiasesBuff<'_> {
        self.layers.biases_buff()
    }
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_> {
        self.layers.biases_buff_mut()
    }

    fn weights_buff(&self) -> Self::WeightsBuff<'_> {
        self.layers.weights_buff()
    }
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_> {
        self.layers.weights_buff_mut()
    }

    fn iter(&self) -> Self::RawParamIter<'_> {
        self.biases_buff().chain(self.weights_buff())
    }
    fn iter_mut(&mut self) -> Self::RawParamIterMut<'_> {
        unsafe {
            let layers_ptr: *mut Layers = &mut self.layers;

            let biases_ref = (&mut *layers_ptr).biases_buff_mut();
            let weights_ref = (&mut *layers_ptr).weights_buff_mut();

            biases_ref.chain(weights_ref)
        }
    }
}

// macro_rules!  {
//     () => {
        
//     };
// }