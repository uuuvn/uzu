use std::{collections::HashMap, mem::ManuallyDrop, range::Range, sync::Arc};

use parking_lot::Mutex;

pub struct BlockAllocator<T, E> {
    min_page_size: usize,
    cache: Mutex<HashMap<usize, Vec<T>>>,
    upstream: Box<dyn Fn(usize) -> Result<T, E> + Send + Sync>,
}

impl<T, E> BlockAllocator<T, E> {
    pub fn new(
        min_page_size: usize,
        upstream: impl Fn(usize) -> Result<T, E> + Send + Sync + 'static,
    ) -> Arc<Self> {
        let min_page_size = min_page_size.clamp(16 * 1024, 16 * 1024); // TODO
        Arc::new(Self {
            min_page_size,
            cache: Mutex::new(HashMap::new()),
            upstream: Box::new(upstream),
        })
    }

    pub fn allocate(
        self: &Arc<Self>,
        size: usize,
    ) -> Result<BlockAllocation<T, E>, E> {
        let page = if let mut cache_locked = self.cache.lock()
            && let Some(cache_entries) = cache_locked.get_mut(&size)
            && let Some(cache_entry) = cache_entries.pop()
        {
            cache_entry
        } else {
            (self.upstream)(size.max(self.min_page_size))?
        };

        Ok(BlockAllocation {
            allocator: self.clone(),
            page: ManuallyDrop::new(page),
            range: (0..size).into(),
        })
    }

    fn free(
        self: &Arc<Self>,
        page: T,
        range: Range<usize>,
    ) {
        self.cache.lock().entry(range.iter().len()).or_default().push(page);
    }
}

pub struct BlockAllocation<T, E> {
    allocator: Arc<BlockAllocator<T, E>>,
    page: ManuallyDrop<T>,
    range: Range<usize>,
}

impl<T, E> BlockAllocation<T, E> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}

impl<T, E> Drop for BlockAllocation<T, E> {
    fn drop(&mut self) {
        self.allocator.free(unsafe { ManuallyDrop::take(&mut self.page) }, self.range);
    }
}
