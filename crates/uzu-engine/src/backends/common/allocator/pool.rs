use std::{collections::VecDeque, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

pub struct PoolAllocator<T, E> {
    pages: [Mutex<VecDeque<T>>; 32],
    upstream: Box<dyn Fn(usize) -> Result<T, E> + Send + Sync>,
}

impl<T, E> PoolAllocator<T, E> {
    pub fn new(upstream: impl Fn(usize) -> Result<T, E> + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            pages: Default::default(),
            upstream: Box::new(upstream),
        })
    }

    pub fn allocate(
        self: &Arc<Self>,
        size: usize,
    ) -> Result<PoolAllocation<T, E>, E> {
        let page = if let Some(page) = self.pages[(size - 1).bit_width() as usize].lock().pop_front() {
            page
        } else {
            (self.upstream)(size.next_power_of_two())?
        };

        Ok(PoolAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    fn alias(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.pages[(range.iter().len() - 1).bit_width() as usize].lock().push_back(page);
    }
}

pub struct PoolAllocation<T, E> {
    allocator: Arc<PoolAllocator<T, E>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T, E> PoolAllocation<T, E> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T, E> Drop for PoolAllocation<T, E> {
    fn drop(&mut self) {
        self.allocator.alias(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}
