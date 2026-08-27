use crate::backends::{
    common::{
        Backend,
        allocator::{
            block::BlockAllocation,
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    metal::{
        buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
        command_buffer::MetalCommandBuffer,
        context::MetalContext,
        error::MetalError,
        kernel::MetalKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type GlobalBuffer = BlockAllocation<MetalDenseBuffer, MetalError>;
    type ConstantBuffer = BumpAllocation<Self::GlobalBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalBuffer, MetalError>;
    type SparseBuffer = MetalSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalBuffer, MetalError>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const NAME: &'static str = "metal";
}
