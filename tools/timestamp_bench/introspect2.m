#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

API_AVAILABLE(macos(26.0))
int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [device newCommandAllocator];
        id<MTL4CommandBuffer> cb = [device newCommandBuffer];
        [cb beginCommandBufferWithAllocator:alloc];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        Class cls = [enc class];

        IMP tsImp = method_getImplementation(class_getInstanceMethod(cls, @selector(writeTimestampWithGranularity:intoHeap:atIndex:)));
        // region covering both precise-only callees (IMP-0x14000 .. IMP+0x1000)
        char name[64];
        snprintf(name, 64, "region_%p.bin", (void*)((char*)tsImp - 0x14000));
        FILE* f = fopen(name, "wb");
        fwrite((char*)tsImp - 0x14000, 1, 0x15000, f);
        fclose(f);
        printf("dumped %s\n", name);

        SEL extra[] = {
            @selector(barrierAfterEncoderStages:beforeEncoderStages:options:),
            @selector(encodeSyncComputeWithBackFacingBarrierSrcMask:BackFacingBarrierDstMask:FrontFacingBarrierSrcMask:FrontFacingBarrierDstMask:),
            @selector(internalResolveCounterHeap:offset:size:destAddress:destResource:),
            @selector(endEncoding),
        };
        const char* files[] = {"agx_barrier.bin", "agx_sync.bin", "agx_resolve.bin", "agx_endencoding.bin"};
        for (int i = 0; i < 4; i++) {
            Method m = class_getInstanceMethod(cls, extra[i]);
            if (!m) { printf("missing %s\n", sel_getName(extra[i])); continue; }
            void* imp = (void*)method_getImplementation(m);
            char fn[80]; snprintf(fn, 80, "%p_%s", imp, files[i]);
            FILE* ff = fopen(fn, "wb");
            fwrite(imp, 1, 8192, ff);
            fclose(ff);
            printf("dumped %s\n", fn);
        }

        // counter heap resolve
        MTL4CounterHeapDescriptor* hd = [MTL4CounterHeapDescriptor new];
        hd.type = MTL4CounterHeapTypeTimestamp; hd.count = 16;
        id<MTL4CounterHeap> heap = [device newCounterHeapWithDescriptor:hd error:nil];
        Class hcls = [heap class];
        Method rm = class_getInstanceMethod(hcls, @selector(resolveCounterRange:));
        if (rm) {
            void* imp = (void*)method_getImplementation(rm);
            char fn[80]; snprintf(fn, 80, "%p_heap_resolve.bin", imp);
            FILE* ff = fopen(fn, "wb"); fwrite(imp, 1, 8192, ff); fclose(ff);
            printf("dumped %s\n", fn);
        }
        [enc endEncoding];
        [cb endCommandBuffer];
    }
    return 0;
}
