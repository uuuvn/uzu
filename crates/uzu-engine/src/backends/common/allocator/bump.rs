use std::{range::Range, sync::Arc};

struct BumpAllocatorPage<T> {
    page: Arc<T>,
    size: usize,
    offset: usize,
}

pub struct BumpAllocator<T, E> {
    min_page_size: usize,
    pages: Vec<BumpAllocatorPage<T>>,
    upstream: Box<dyn Fn(usize) -> Result<T, E> + Send + Sync>,
}

impl<T, E> BumpAllocator<T, E> {
    pub fn new(
        min_page_size: usize,
        upstream: impl Fn(usize) -> Result<T, E> + Send + Sync + 'static,
    ) -> Self {
        Self {
            min_page_size,
            pages: Vec::new(),
            upstream: Box::new(upstream),
        }
    }

    pub fn allocate(
        &mut self,
        size: usize,
    ) -> Result<BumpAllocation<T>, E> {
        let alignment = usize::min(size, 64).next_power_of_two();

        if let Some(page) = self.pages.last_mut()
            && let page_offset_aligned = page.offset.next_multiple_of(alignment)
            && page_offset_aligned + size <= page.size
        {
            let allocation = BumpAllocation {
                page: page.page.clone(),
                range: (page_offset_aligned..page_offset_aligned + size).into(),
            };
            page.offset = page_offset_aligned + size;
            return Ok(allocation);
        }

        let new_page_size = usize::max(size, self.min_page_size);
        let new_page = Arc::new((self.upstream)(new_page_size)?);
        self.pages.push(BumpAllocatorPage {
            page: new_page.clone(),
            size: new_page_size,
            offset: size,
        });

        Ok(BumpAllocation {
            page: new_page,
            range: (0..size).into(),
        })
    }

    pub fn is_done(&self) -> bool {
        self.pages.iter().all(|page| Arc::strong_count(&page.page) == 1)
    }
}

pub struct BumpAllocation<T> {
    page: Arc<T>,
    range: Range<usize>,
}

impl<T> BumpAllocation<T> {
    pub fn page(&self) -> &T {
        &self.page
    }

    pub fn range(&self) -> Range<usize> {
        self.range
    }
}
